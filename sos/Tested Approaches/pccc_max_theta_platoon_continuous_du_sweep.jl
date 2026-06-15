using JuMP
using JSON
using Printf
const MOI = JuMP.MOI
try
    @eval using MosekTools
catch err
    error("MosekTools is required. Original error: $(err)")
end

const D_SAFE = 0.3
D_UNSAFE = 0.4
const N_REGIONS = 3
const NBASIS = 6
const X1_MIN = 0.0
const X1_MAX = 3.0
const X2_MIN = 0.0
const X2_MAX = 5.1
const NODES = [1, 2]
const MODES = [1, 2]
const EDGES = [(1,1,1), (1,1,2), (1,2,2), (2,2,1)]
const RANK_NODES = [1]

function f_mode(sigma::Int, x::Vector{Float64})
    x1, x2 = x
    if sigma == 1
        return [0.01*x2 + 0.9*x1 - 0.02*x1^2, 2.0 + 0.8*x2 - 0.04*x2^2]
    elseif sigma == 2
        return [0.9*x1 - 0.02*x1^2, 2.0 + 0.8*x2 - 0.04*x2^2]
    else
        error("unknown mode")
    end
end

function quad_basis(x::Vector{Float64})
    x1, x2 = x
    return [1.0, x1, x2, x1^2, x1*x2, x2^2]
end

function region_index(x::Vector{Float64})
    gap = x[2] - x[1]
    if gap <= D_SAFE + 1e-12
        return 1
    elseif gap <= D_UNSAFE + 1e-12
        return 2
    else
        return 3
    end
end

function make_grid(n1::Int, n2::Int)
    pts = Vector{Vector{Float64}}()
    for x1 in range(X1_MIN, X1_MAX; length=n1), x2 in range(X2_MIN, X2_MAX; length=n2)
        if x2 + 1e-12 >= x1
            push!(pts, [Float64(x1), Float64(x2)])
        end
    end
    return pts
end

function make_X0_grid(n1::Int, n2::Int)
    raw = make_grid(n1, n2)
    return [x for x in raw if x[2] - x[1] <= D_SAFE + 1e-12]
end

function make_Xu_grid(n1::Int, n2::Int)
    raw = make_grid(n1, n2)
    return [x for x in raw if x[2] - x[1] <= D_UNSAFE + 1e-12]
end

function remove_X0_from_Xu(Xu, X0)
    out = Vector{Vector{Float64}}()
    for x in Xu
        in_x0 = any(maximum(abs.(x .- x0)) <= 1e-10 for x0 in X0)
        if !in_x0
            push!(out, x)
        end
    end
    return out
end

function V_expr(c, v::Int, x::Vector{Float64})
    r = region_index(x)
    b = quad_basis(x)
    return sum(c[v,r,k] * b[k] for k in 1:NBASIS)
end

function C_expr(c, v::Int, x::Vector{Float64}, xp::Vector{Float64})
    return V_expr(c, v, xp) - V_expr(c, v, x)
end

# Exact continuity constraints for piecewise-quadratic potentials across gap boundaries.
# For boundary gap = d, enforce V_v^left(t,t+d) == V_v^right(t,t+d)
# by matching constant, linear, and quadratic coefficients in t=x1.
function add_gap_continuity_constraints!(model, c)
    for v in NODES
        for (r_left, r_right, dgap) in [(1, 2, D_SAFE), (2, 3, D_UNSAFE)]
            a(k) = c[v, r_left, k] - c[v, r_right, k]
            @constraint(model, a(1) + dgap * a(3) + dgap^2 * a(6) == 0.0)
            @constraint(model, a(2) + a(3) + dgap * a(5) + 2.0 * dgap * a(6) == 0.0)
            @constraint(model, a(4) + a(5) + a(6) == 0.0)
        end
    end
end

function solve_max_theta(label::String; use_p1::Bool=true, use_p2::Bool=true, use_p3::Bool=true,
                         zeta::Float64=3.0, lambda1::Float64=0.95,
                         lambda2::Float64=0.02, lambda3::Float64=0.02,
                         norm_value::Float64=-1.0,
                         coarse_n1::Int=7, coarse_n2::Int=7,
                         wf_n1::Int=7, wf_n2::Int=41,
                         x0_n1::Int=7, x0_n2::Int=41,
                         coeff_bound::Float64=100.0,
                         theta_lb::Float64=-1.0, theta_ub::Float64=1.0,
                         continuous::Bool=true)
    X = make_grid(coarse_n1, coarse_n2)
    Xp = X
    X0 = make_X0_grid(x0_n1, x0_n2)
    Xu = make_Xu_grid(wf_n1, wf_n2)
    Xu_wf = remove_X0_from_Xu(Xu, X0)
    if use_p3 && (isempty(X0) || isempty(Xu_wf))
        return Dict("label"=>label, "term"=>"EMPTY_P3_GRID", "primal"=>"NO_SOLUTION", "theta_max"=>nothing, "rho"=>nothing, "num_X"=>length(X), "num_X0"=>length(X0), "num_Xu_wf"=>length(Xu_wf), "coeff_bound"=>coeff_bound)
    end

    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, c[v in NODES, r in 1:N_REGIONS, k in 1:NBASIS])
    @variable(model, 0.0 <= rho <= coeff_bound)
    @variable(model, theta_lb <= theta <= theta_ub)
    for v in NODES, r in 1:N_REGIONS, k in 1:NBASIS
        @constraint(model, c[v,r,k] <= rho)
        @constraint(model, -c[v,r,k] <= rho)
    end

    if continuous
        add_gap_continuity_constraints!(model, c)
    end

    if use_p1
        for v in NODES, x in X, sigma in MODES
            @constraint(model, C_expr(c, v, x, f_mode(sigma, x)) <= zeta)
        end
    end

    if use_p2
        for (u,sigma,v) in EDGES, x in X
            xnext = f_mode(sigma, x)
            for xp in Xp
                @constraint(model, C_expr(c, u, x, xp) <= lambda1*C_expr(c, v, xnext, xp) + (1.0-lambda1)*zeta)
            end
        end
    end

    if use_p3
        for v in RANK_NODES, x0 in X0, x in Xu_wf, xp in Xu_wf
            @constraint(model,
                C_expr(c, v, x0, xp) + C_expr(c, v, x, x0) -
                lambda2*C_expr(c, v, x0, x) - lambda3*C_expr(c, v, x, xp) +
                (theta + lambda2*zeta + lambda3*zeta) <= 0.0)
        end
    end

    x_ref = X[1]
    xp_ref = Xp[end]
    @constraint(model, C_expr(c, 1, x_ref, xp_ref) == norm_value)
    @objective(model, Max, theta)
    optimize!(model)
    term = string(termination_status(model))
    primal = string(primal_status(model))
    good = primal_status(model) in [MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT]
    return Dict("label"=>label, "term"=>term, "primal"=>primal,
                "theta_max"=>good ? value(theta) : nothing,
                "rho"=>good ? value(rho) : nothing,
                "num_X"=>length(X), "num_X0"=>length(X0), "num_Xu_wf"=>length(Xu_wf),
                "use_p1"=>use_p1, "use_p2"=>use_p2, "use_p3"=>use_p3,
                "zeta"=>zeta, "lambda1"=>lambda1, "lambda2"=>lambda2,
                "lambda3"=>lambda3, "norm_value"=>norm_value,
                "coeff_bound"=>coeff_bound, "theta_lb"=>theta_lb, "theta_ub"=>theta_ub, "continuous"=>continuous)
end

function print_row(r)
    th = r["theta_max"] === nothing ? "-" : @sprintf("%.9e", r["theta_max"])
    rho = r["rho"] === nothing ? "-" : @sprintf("%.3e", r["rho"])
    println(rpad(r["label"], 42), "  ", lpad(@sprintf("%.1e", r["coeff_bound"]), 9), "  ",
            rpad(r["term"], 14), "  ", rpad(r["primal"], 18),
            "  theta_max=", th, "  rho=", rho, "  X0=", r["num_X0"], "  Xu_wf=", r["num_Xu_wf"])
end

function main()
    global D_UNSAFE
    coeff_bounds = [10000.0, 100000.0]
    dunsafe_values = [0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.00]
    cases = Dict{String,Any}[]

    # Main proof-oriented setting: dense non-vacuous WF grid.
    # If this is too expensive, first temporarily change wf/x0 to 15,41 for scouting,
    # but final go/no-go should use 21,41.
    wf1, wf2 = 21, 41
    x01, x02 = 21, 41

    for du in dunsafe_values
        D_UNSAFE = du
        for rb in coeff_bounds
            label = @sprintf("DU=%.2f coarse15 wf%dx%d x0%dx%d", du, wf1, wf2, x01, x02)
            push!(cases, solve_max_theta(label,
                coarse_n1=15, coarse_n2=15,
                wf_n1=wf1, wf_n2=wf2, x0_n1=x01, x0_n2=x02,
                coeff_bound=rb, lambda2=0.02, lambda3=0.02,
                theta_lb=-1.0, theta_ub=1.0))
        end
    end

    println("Continuous pwquad max-theta D_UNSAFE/rho sweep")
    println("Forward grid: coarse 15x15; WF/X0 grid: 21x41; original lambdas lambda2=lambda3=0.02; continuity=true")
    println("D_SAFE = ", D_SAFE)
    println("label                                       rho_bound  term            primal              theta/rho/counts")
    println("-----------------------------------------------------------------------------------------------------------")
    for r in cases
        print_row(r)
    end
    mkpath(joinpath("results", "diagnostics"))
    out = joinpath("results", "diagnostics", "platoon_continuous_du_sweep.json")
    open(out, "w") do io
        JSON.print(io, cases, 2)
    end
    println("Saved D_UNSAFE sweep diagnostic JSON: ", out)
end

main()
