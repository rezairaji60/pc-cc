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

function gap(x::Vector{Float64})
    return x[2] - x[1]
end

function region_index(x::Vector{Float64})
    g = gap(x)
    if g <= D_SAFE + 1e-12
        return 1
    elseif g <= D_UNSAFE + 1e-12
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
    return [x for x in raw if gap(x) <= D_SAFE + 1e-12]
end

function make_Xu_grid(n1::Int, n2::Int)
    raw = make_grid(n1, n2)
    return [x for x in raw if gap(x) <= D_UNSAFE + 1e-12]
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

function V_eval(cval, v::Int, x::Vector{Float64})
    r = region_index(x)
    b = quad_basis(x)
    return sum(cval[v,r,k] * b[k] for k in 1:NBASIS)
end

function C_eval(cval, v::Int, x::Vector{Float64}, xp::Vector{Float64})
    return V_eval(cval, v, xp) - V_eval(cval, v, x)
end

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

function distance_to_successor(xp::Vector{Float64}, x::Vector{Float64})
    dmin = Inf
    arg = 0
    for sigma in MODES
        fx = f_mode(sigma, x)
        d = maximum(abs.(xp .- fx))
        if d < dmin
            dmin = d
            arg = sigma
        end
    end
    return dmin, arg
end

function p3_residual(cval, v, x0, x, xp, theta, zeta, lambda2, lambda3)
    c_x0_xp = C_eval(cval, v, x0, xp)
    c_x_x0  = C_eval(cval, v, x,  x0)
    c_x0_x  = C_eval(cval, v, x0, x)
    c_x_xp  = C_eval(cval, v, x,  xp)
    lhs = c_x0_xp + c_x_x0 - lambda2*c_x0_x - lambda3*c_x_xp + (theta + lambda2*zeta + lambda3*zeta)
    base = c_x0_xp + c_x_x0 - lambda2*c_x0_x - lambda3*c_x_xp + (lambda2*zeta + lambda3*zeta)
    return lhs, base, c_x0_xp, c_x_x0, c_x0_x, c_x_xp
end

function solve_and_inspect(; du::Float64, coeff_bound::Float64=100000.0,
                           coarse_n1::Int=15, coarse_n2::Int=15,
                           wf_n1::Int=21, wf_n2::Int=41,
                           x0_n1::Int=21, x0_n2::Int=41,
                           zeta::Float64=3.0, lambda1::Float64=0.95,
                           lambda2::Float64=0.02, lambda3::Float64=0.02,
                           theta_lb::Float64=-1.0, theta_ub::Float64=1.0)
    global D_UNSAFE
    D_UNSAFE = du
    X = make_grid(coarse_n1, coarse_n2)
    Xp = X
    X0 = make_X0_grid(x0_n1, x0_n2)
    Xu = make_Xu_grid(wf_n1, wf_n2)
    Xu_wf = remove_X0_from_Xu(Xu, X0)
    if isempty(X0) || isempty(Xu_wf)
        return Dict("D_UNSAFE"=>du, "term"=>"EMPTY_P3_GRID", "primal"=>"NO_SOLUTION", "theta_max"=>nothing,
                    "num_X0"=>length(X0), "num_Xu_wf"=>length(Xu_wf))
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
    add_gap_continuity_constraints!(model, c)

    for v in NODES, x in X, sigma in MODES
        @constraint(model, C_expr(c, v, x, f_mode(sigma, x)) <= zeta)
    end

    for (u,sigma,v) in EDGES, x in X
        xnext = f_mode(sigma, x)
        for xp in Xp
            @constraint(model, C_expr(c, u, x, xp) <= lambda1*C_expr(c, v, xnext, xp) + (1.0-lambda1)*zeta)
        end
    end

    for v in RANK_NODES, x0 in X0, x in Xu_wf, xp in Xu_wf
        @constraint(model,
            C_expr(c, v, x0, xp) + C_expr(c, v, x, x0) -
            lambda2*C_expr(c, v, x0, x) - lambda3*C_expr(c, v, x, xp) +
            (theta + lambda2*zeta + lambda3*zeta) <= 0.0)
    end

    @constraint(model, C_expr(c, 1, X[1], Xp[end]) == -1.0)
    @objective(model, Max, theta)
    optimize!(model)

    term = string(termination_status(model))
    primal = string(primal_status(model))
    good = primal_status(model) in [MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT]
    out = Dict{String,Any}(
        "D_UNSAFE"=>du, "term"=>term, "primal"=>primal, "num_X"=>length(X), "num_X0"=>length(X0),
        "num_Xu_wf"=>length(Xu_wf), "coeff_bound"=>coeff_bound, "zeta"=>zeta,
        "lambda1"=>lambda1, "lambda2"=>lambda2, "lambda3"=>lambda3,
        "theta_max"=>good ? value(theta) : nothing, "rho"=>good ? value(rho) : nothing)

    if !good
        return out
    end

    cval = zeros(Float64, maximum(NODES), N_REGIONS, NBASIS)
    for v in NODES, r in 1:N_REGIONS, k in 1:NBASIS
        cval[v,r,k] = value(c[v,r,k])
    end
    th = value(theta)

    worst = Dict{String,Any}()
    maxlhs = -Inf
    for v in RANK_NODES, x0 in X0, x in Xu_wf, xp in Xu_wf
        lhs, base, c_x0_xp, c_x_x0, c_x0_x, c_x_xp = p3_residual(cval, v, x0, x, xp, th, zeta, lambda2, lambda3)
        if lhs > maxlhs
            dsucc, sigma_near = distance_to_successor(xp, x)
            maxlhs = lhs
            worst = Dict(
                "node"=>v,
                "x0"=>x0, "x"=>x, "xp"=>xp,
                "gap_x0"=>gap(x0), "gap_x"=>gap(x), "gap_xp"=>gap(xp),
                "region_x0"=>region_index(x0), "region_x"=>region_index(x), "region_xp"=>region_index(xp),
                "C_x0_xp"=>c_x0_xp, "C_x_x0"=>c_x_x0, "C_x0_x"=>c_x0_x, "C_x_xp"=>c_x_xp,
                "lambda2_C_x0_x"=>lambda2*c_x0_x, "lambda3_C_x_xp"=>lambda3*c_x_xp,
                "lambda2zeta_plus_lambda3zeta"=>lambda2*zeta + lambda3*zeta,
                "theta"=>th, "base_without_theta"=>base, "lhs_at_theta"=>lhs,
                "is_diagonal_x_xp"=>maximum(abs.(x .- xp)) <= 1e-10,
                "distance_xp_to_nearest_successor_of_x"=>dsucc,
                "nearest_successor_mode"=>sigma_near,
                "x_near_D_SAFE"=>abs(gap(x)-D_SAFE) <= 1e-10,
                "xp_near_D_SAFE"=>abs(gap(xp)-D_SAFE) <= 1e-10,
                "x_near_D_UNSAFE"=>abs(gap(x)-D_UNSAFE) <= 1e-10,
                "xp_near_D_UNSAFE"=>abs(gap(xp)-D_UNSAFE) <= 1e-10)
        end
    end
    out["worst_p3"] = worst
    out["max_p3_lhs_at_theta"] = maxlhs
    return out
end

function fmtvec(x)
    return @sprintf("(%.10g, %.10g)", x[1], x[2])
end

function print_case(r)
    println("\n============================================================")
    @printf("D_UNSAFE = %.3f\n", r["D_UNSAFE"])
    println("term = ", r["term"], ", primal = ", r["primal"])
    println("|X| = ", r["num_X"], ", |X0| = ", r["num_X0"], ", |Xu_wf| = ", r["num_Xu_wf"])
    if r["theta_max"] === nothing
        println("No usable primal solution.")
        return
    end
    @printf("theta_max = %.12e, rho = %.6e\n", r["theta_max"], r["rho"])
    w = r["worst_p3"]
    println("Worst P3 constraint at max-theta solution:")
    println("  node v = ", w["node"])
    println("  x0 = ", fmtvec(w["x0"]), "  gap=", @sprintf("%.10g", w["gap_x0"]), "  region=", w["region_x0"])
    println("  x  = ", fmtvec(w["x"]),  "  gap=", @sprintf("%.10g", w["gap_x"]),  "  region=", w["region_x"])
    println("  xp = ", fmtvec(w["xp"]), "  gap=", @sprintf("%.10g", w["gap_xp"]), "  region=", w["region_xp"])
    println("  is_diagonal_x_xp = ", w["is_diagonal_x_xp"])
    println("  dist(xp, nearest f_sigma(x)) = ", @sprintf("%.6e", w["distance_xp_to_nearest_successor_of_x"]), "  mode=", w["nearest_successor_mode"])
    println("  boundary flags: x near D_SAFE=", w["x_near_D_SAFE"], ", xp near D_SAFE=", w["xp_near_D_SAFE"],
            ", x near D_UNSAFE=", w["x_near_D_UNSAFE"], ", xp near D_UNSAFE=", w["xp_near_D_UNSAFE"])
    println("  C(x0,xp) = ", @sprintf("%.12e", w["C_x0_xp"]))
    println("  C(x,x0)  = ", @sprintf("%.12e", w["C_x_x0"]))
    println("  C(x0,x)  = ", @sprintf("%.12e", w["C_x0_x"]))
    println("  C(x,xp)  = ", @sprintf("%.12e", w["C_x_xp"]))
    println("  lambda2*C(x0,x) = ", @sprintf("%.12e", w["lambda2_C_x0_x"]))
    println("  lambda3*C(x,xp) = ", @sprintf("%.12e", w["lambda3_C_x_xp"]))
    println("  lambda2*zeta + lambda3*zeta = ", @sprintf("%.12e", w["lambda2zeta_plus_lambda3zeta"]))
    println("  base without theta = ", @sprintf("%.12e", w["base_without_theta"]))
    println("  lhs at theta_max = ", @sprintf("%.12e", w["lhs_at_theta"]))
end

function main()
    results = Dict{String,Any}[]
    for du in [0.35, 0.40, 0.50, 1.00]
        r = solve_and_inspect(du=du, coeff_bound=100000.0,
                              coarse_n1=15, coarse_n2=15,
                              wf_n1=21, wf_n2=41,
                              x0_n1=21, x0_n2=41)
        push!(results, r)
        print_case(r)
    end
    mkpath(joinpath("results", "diagnostics"))
    out = joinpath("results", "diagnostics", "p3_worst_constraint_audit.json")
    open(out, "w") do io
        JSON.print(io, results, 2)
    end
    println("\nSaved P3 worst-constraint audit JSON: ", out)
end

main()
