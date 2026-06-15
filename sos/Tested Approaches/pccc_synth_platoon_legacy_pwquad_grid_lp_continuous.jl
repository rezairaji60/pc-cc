# Reproduce the legacy piecewise-quadratic V-based PC-CC grid synthesis in Julia.
#
# Certificate structure:
#   C_v(x, xp) = V_v(xp) - V_v(x)
#   V_v is quadratic in x on each of three gap regions.
#
# This is a finite-grid convex QP/LP-style synthesis, not an SOS proof.
# It is intended as the first reproducibility step before any SOS/TSSOS lift.

using JuMP
using JSON
using LinearAlgebra

const MOI = JuMP.MOI

# Try MOSEK first. If not available, fail with a clear message.
try
    @eval using MosekTools
catch err
    error("MosekTools is required for this script. Install/add MosekTools or adapt optimizer selection. Original error: $(err)")
end

# -----------------------------------------------------------------------------
# Hyperparameters matching the legacy Python script
# -----------------------------------------------------------------------------
const ZETA = 3.0
const LAMBDA1 = 0.95
const LAMBDA2 = 0.02
const LAMBDA3 = 0.02
const THETA = 5.0e-5

const DELTA1 = 0.0
const DELTA2 = 0.0
const DELTA3 = 0.0

const D_SAFE = 0.3
const D_UNSAFE = 0.4
const N_REGIONS = 3
const NBASIS = 6

const X1_MIN = 0.0
const X1_MAX = 3.0
const X2_MIN = 0.0
const X2_MAX = 5.1

# Default synthesis / validation grids.
# The original legacy Python used WF_N1=WF_N2=5 and X0_N1=X0_N2=5,
# but on this platoon domain that makes Xu_minus_X0 empty, so condition (3) is vacuous.
# These defaults use the same 21 x 41 grid as the dense validator for X0/Xu.
const COARSE_N1_DEFAULT = 7
const COARSE_N2_DEFAULT = 7
const FINE_N1_DEFAULT = 9
const FINE_N2_DEFAULT = 9
const WF_N1_DEFAULT = 21
const WF_N2_DEFAULT = 41
const X0_N1_DEFAULT = 21
const X0_N2_DEFAULT = 41

const NUM_TOL = 1.0e-8

# 1-based nodes/modes/edges. This is the same graph as the 0-based legacy file.
const NODES = [1, 2]
const MODES = [1, 2]
const EDGES = [
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 2),
    (2, 2, 1),
]

# Legacy ranking node [0] becomes node 1 here.
const RANK_NODES = [1]

# -----------------------------------------------------------------------------
# Dynamics and bases
# -----------------------------------------------------------------------------
function f_mode(sigma::Int, x::Vector{Float64})
    x1, x2 = x
    if sigma == 1
        return [0.01 * x2 + 0.9 * x1 - 0.02 * x1^2,
                2.0 + 0.8 * x2 - 0.04 * x2^2]
    elseif sigma == 2
        return [0.9 * x1 - 0.02 * x1^2,
                2.0 + 0.8 * x2 - 0.04 * x2^2]
    else
        error("unknown mode $(sigma)")
    end
end

function quad_basis(x::Vector{Float64})
    x1, x2 = x
    return [1.0, x1, x2, x1^2, x1 * x2, x2^2]
end

function region_index(x::Vector{Float64})
    gap = x[2] - x[1]
    if gap <= D_SAFE + 1.0e-12
        return 1
    elseif gap <= D_UNSAFE + 1.0e-12
        return 2
    else
        return 3
    end
end

function make_grid(n1::Int, n2::Int)
    pts = Vector{Vector{Float64}}()
    x1_vals = collect(range(X1_MIN, X1_MAX; length=n1))
    x2_vals = collect(range(X2_MIN, X2_MAX; length=n2))
    for x1 in x1_vals, x2 in x2_vals
        if x2 + 1.0e-12 >= x1
            push!(pts, [Float64(x1), Float64(x2)])
        end
    end
    return pts, x1_vals, x2_vals
end

function make_X0_grid(n1::Int, n2::Int)
    raw, _, _ = make_grid(n1, n2)
    return [x for x in raw if x[2] - x[1] <= D_SAFE + 1.0e-12]
end

function make_Xu_grid(n1::Int, n2::Int)
    raw, _, _ = make_grid(n1, n2)
    return [x for x in raw if x[2] - x[1] <= D_UNSAFE + 1.0e-12]
end

function remove_X0_from_Xu(Xu::Vector{Vector{Float64}}, X0::Vector{Vector{Float64}})
    out = Vector{Vector{Float64}}()
    for x in Xu
        in_x0 = any(norm(x .- x0, Inf) <= 1.0e-10 for x0 in X0)
        if !in_x0
            push!(out, x)
        end
    end
    return out
end

function V_expr(c, v::Int, x::Vector{Float64})
    r = region_index(x)
    b = quad_basis(x)
    return sum(c[v, r, k] * b[k] for k in 1:NBASIS)
end

function C_expr(c, v::Int, x::Vector{Float64}, xp::Vector{Float64})
    return V_expr(c, v, xp) - V_expr(c, v, x)
end

function V_eval(cval, v::Int, x::Vector{Float64})
    r = region_index(x)
    b = quad_basis(x)
    return sum(cval[v][r][k] * b[k] for k in 1:NBASIS)
end

function C_eval(cval, v::Int, x::Vector{Float64}, xp::Vector{Float64})
    return V_eval(cval, v, xp) - V_eval(cval, v, x)
end

function coeff_array_from_model(c)
    return [[[value(c[v, r, k]) for k in 1:NBASIS] for r in 1:N_REGIONS] for v in NODES]
end


# Exact continuity constraints for piecewise-quadratic potentials across gap boundaries.
# For boundary gap = d, enforce V_v^left(t,t+d) == V_v^right(t,t+d)
# as a polynomial identity in t=x1, i.e., match constant, linear, quadratic coefficients.
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

# -----------------------------------------------------------------------------
# Validation of the legacy inequalities on specified grids
# -----------------------------------------------------------------------------
function legacy_metrics(cval; fine_n1::Int=FINE_N1_DEFAULT, fine_n2::Int=FINE_N2_DEFAULT,
                        wf_n1::Int=WF_N1_DEFAULT, wf_n2::Int=WF_N2_DEFAULT,
                        x0_n1::Int=X0_N1_DEFAULT, x0_n2::Int=X0_N2_DEFAULT)
    Xfine, x1_vals, x2_vals = make_grid(fine_n1, fine_n2)
    Xpfine = Xfine
    X0 = make_X0_grid(x0_n1, x0_n2)
    Xu = make_Xu_grid(wf_n1, wf_n2)
    Xu_wf = remove_X0_from_Xu(Xu, X0)

    viol1 = -Inf
    viol2 = -Inf
    viol3 = -Inf
    worst1 = nothing
    worst2 = nothing
    worst3 = nothing

    # (1) C_v(x,f_sigma(x)) <= zeta
    for v in NODES, x in Xfine, sigma in MODES
        xp = f_mode(sigma, x)
        lhs = C_eval(cval, v, x, xp) - (ZETA - DELTA1)
        if lhs > viol1
            viol1 = lhs
            worst1 = Dict("v"=>v, "sigma"=>sigma, "x"=>x, "fx"=>xp)
        end
    end

    # (2) C_u(x,xp) <= lambda1*C_v(f_sigma(x),xp)+(1-lambda1)*zeta
    for (u, sigma, v) in EDGES, x in Xfine
        xnext = f_mode(sigma, x)
        for xp in Xpfine
            lhs = C_eval(cval, u, x, xp) - (LAMBDA1 * C_eval(cval, v, xnext, xp) + (1.0 - LAMBDA1) * ZETA - DELTA2)
            if lhs > viol2
                viol2 = lhs
                worst2 = Dict("edge"=>[u,sigma,v], "x"=>x, "fx"=>xnext, "xp"=>xp)
            end
        end
    end

    # (3) Well-foundedness on rank nodes only.
    if isempty(X0) || isempty(Xu_wf)
        error("Cannot synthesize legacy condition (3): |X0|=$(length(X0)), |Xu_minus_X0|=$(length(Xu_wf)). Increase wf_n1/wf_n2 and x0_n1/x0_n2.")
    end
    for v in RANK_NODES, x0 in X0, x in Xu_wf, xp in Xu_wf
            lhs = C_eval(cval, v, x0, xp) + C_eval(cval, v, x, x0) -
                  LAMBDA2 * C_eval(cval, v, x0, x) -
                  LAMBDA3 * C_eval(cval, v, x, xp) +
                  (THETA + LAMBDA2 * ZETA + LAMBDA3 * ZETA) + DELTA3
            if lhs > viol3
                viol3 = lhs
                worst3 = Dict("v"=>v, "x0"=>x0, "x"=>x, "xp"=>xp)
            end
        end

    h1 = length(x1_vals) > 1 ? Float64(x1_vals[2] - x1_vals[1]) : 0.0
    h2 = length(x2_vals) > 1 ? Float64(x2_vals[2] - x2_vals[1]) : 0.0

    return Dict(
        "viol1_transition_boundedness" => viol1,
        "viol2_forward_contraction" => viol2,
        "viol3_well_foundedness" => viol3,
        "grid_certified" => (viol1 <= NUM_TOL && viol2 <= NUM_TOL && viol3 <= NUM_TOL),
        "fine_n1" => fine_n1,
        "fine_n2" => fine_n2,
        "wf_n1" => wf_n1,
        "wf_n2" => wf_n2,
        "x0_n1" => x0_n1,
        "x0_n2" => x0_n2,
        "num_X_fine" => length(Xfine),
        "num_X0" => length(X0),
        "num_Xu_wf" => length(Xu_wf),
        "h_x1" => h1,
        "h_x2" => h2,
        "worst1" => worst1,
        "worst2" => worst2,
        "worst3" => worst3,
    )
end

function solve_legacy_grid(; coarse_n1::Int=COARSE_N1_DEFAULT, coarse_n2::Int=COARSE_N2_DEFAULT,
                           wf_n1::Int=WF_N1_DEFAULT, wf_n2::Int=WF_N2_DEFAULT,
                           x0_n1::Int=X0_N1_DEFAULT, x0_n2::Int=X0_N2_DEFAULT,
                           fine_n1::Int=FINE_N1_DEFAULT, fine_n2::Int=FINE_N2_DEFAULT,
                           coeff_bound::Float64=1.0e4)
    X, x1_vals, x2_vals = make_grid(coarse_n1, coarse_n2)
    Xp = X
    X0 = make_X0_grid(x0_n1, x0_n2)
    Xu = make_Xu_grid(wf_n1, wf_n2)
    Xu_wf = remove_X0_from_Xu(Xu, X0)

    println("Legacy V-based continuous piecewise-quadratic grid synthesis")
    println("  coarse grid |X| = ", length(X), "  coarse_n1=", coarse_n1, " coarse_n2=", coarse_n2)
    println("  |X0| = ", length(X0), "  |Xu_wf| = ", length(Xu_wf))
    println("  graph edges = ", EDGES)
    println("  coeff_bound = ", coeff_bound)

    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, c[v in NODES, r in 1:N_REGIONS, k in 1:NBASIS])
    @variable(model, rho >= 0.0)
    for v in NODES, r in 1:N_REGIONS, k in 1:NBASIS
        @constraint(model, c[v,r,k] <= rho)
        @constraint(model, -c[v,r,k] <= rho)
    end
    @constraint(model, rho <= coeff_bound)

    # Continuity across gap-region boundaries gap = D_SAFE and gap = D_UNSAFE.
    add_gap_continuity_constraints!(model, c)

    # (1) Transition boundedness.
    for v in NODES, x in X, sigma in MODES
        xp = f_mode(sigma, x)
        @constraint(model, C_expr(c, v, x, xp) <= ZETA - DELTA1)
    end

    # (2) Forward contraction along edges.
    for (u, sigma, v) in EDGES, x in X
        xnext = f_mode(sigma, x)
        for xp in Xp
            @constraint(model,
                C_expr(c, u, x, xp) <=
                LAMBDA1 * C_expr(c, v, xnext, xp) + (1.0 - LAMBDA1) * ZETA - DELTA2)
        end
    end

    # (3) Well-foundedness / finite visits of Xu, rank node only.
    if isempty(X0) || isempty(Xu_wf)
        error("Cannot synthesize legacy condition (3): |X0|=$(length(X0)), |Xu_minus_X0|=$(length(Xu_wf)). Increase wf_n1/wf_n2 and x0_n1/x0_n2.")
    end
    for v in RANK_NODES, x0 in X0, x in Xu_wf, xp in Xu_wf
            @constraint(model,
                C_expr(c, v, x0, xp) + C_expr(c, v, x, x0) -
                LAMBDA2 * C_expr(c, v, x0, x) -
                LAMBDA3 * C_expr(c, v, x, xp) +
                (THETA + LAMBDA2 * ZETA + LAMBDA3 * ZETA) + DELTA3 <= 0.0)
        end

    # Same normalization as legacy Python: C_1(x_ref,xp_ref) = -1.
    x_ref = X[1]
    xp_ref = Xp[end]
    @constraint(model, C_expr(c, 1, x_ref, xp_ref) == -1.0)

    # Linear feasibility/regularization objective. Avoids conic-QP numerical issues.
    @objective(model, Min, rho)

    println("  constraints = ", num_constraints(model; count_variable_in_set_constraints=false))
    println("Starting MOSEK optimization...")
    optimize!(model)

    term = string(termination_status(model))
    primal = string(primal_status(model))
    println("Termination status = ", term)
    println("Primal status      = ", primal)

    if !(termination_status(model) in [MOI.OPTIMAL, MOI.SLOW_PROGRESS] && primal_status(model) in [MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT])
        error("No usable primal solution. term=$(term), primal=$(primal)")
    end

    cval = coeff_array_from_model(c)
    obj = objective_value(model)
    println("Objective value = ", obj, "  (rho coefficient bound)")

    metrics = legacy_metrics(cval; fine_n1=fine_n1, fine_n2=fine_n2,
                             wf_n1=wf_n1, wf_n2=wf_n2,
                             x0_n1=x0_n1, x0_n2=x0_n2)
    println("Validation on check grids:")
    println("  viol1 transition boundedness = ", metrics["viol1_transition_boundedness"])
    println("  viol2 forward contraction    = ", metrics["viol2_forward_contraction"])
    println("  viol3 well-foundedness       = ", metrics["viol3_well_foundedness"])
    println("  grid certified = ", metrics["grid_certified"])

    h1 = length(x1_vals) > 1 ? Float64(x1_vals[2] - x1_vals[1]) : 0.0
    h2 = length(x2_vals) > 1 ? Float64(x2_vals[2] - x2_vals[1]) : 0.0

    return Dict(
        "certificate_type" => "legacy_piecewise_quadratic_node_potential_grid_continuous",
        "description" => "Julia reproduction of legacy V-based PC-CC grid synthesis with exact continuity constraints across gap-region boundaries",
        "status" => term,
        "primal_status" => primal,
        "objective" => obj,
        "objective_description" => "minimize rho subject to |coefficients| <= rho <= coeff_bound",
        "coeff_bound" => coeff_bound,
        "grid_certified" => metrics["grid_certified"],
        "zeta" => ZETA,
        "lambda1" => LAMBDA1,
        "lambda2" => LAMBDA2,
        "lambda3" => LAMBDA3,
        "theta" => THETA,
        "delta1" => DELTA1,
        "delta2" => DELTA2,
        "delta3" => DELTA3,
        "d_safe" => D_SAFE,
        "d_unsafe" => D_UNSAFE,
        "nodes" => NODES,
        "edges" => [[a,b,c] for (a,b,c) in EDGES],
        "rank_nodes" => RANK_NODES,
        "n_regions" => N_REGIONS,
        "basis" => ["1", "x1", "x2", "x1^2", "x1*x2", "x2^2"],
        "region_description" => "Region 1: gap<=D_SAFE; Region 2: D_SAFE<gap<=D_UNSAFE; Region 3: gap>D_UNSAFE",
        "continuity_constraints" => true,
        "continuity_boundaries" => [D_SAFE, D_UNSAFE],
        "coeffs_reg" => cval,
        "synthesis_grid" => Dict("coarse_n1"=>coarse_n1, "coarse_n2"=>coarse_n2, "num_X"=>length(X), "h_x1"=>h1, "h_x2"=>h2, "x0_n1"=>x0_n1, "x0_n2"=>x0_n2, "wf_n1"=>wf_n1, "wf_n2"=>wf_n2, "num_X0"=>length(X0), "num_Xu_wf"=>length(Xu_wf)),
        "validation_metrics" => metrics,
        "note" => "Finite-grid certificate only; not an SOS proof. Includes exact continuity constraints for V_v across gap boundaries. Use for candidate discovery before SOS/TSSOS lift.",
    )
end

function sanitize_json(x)
    if x isa AbstractFloat
        if isfinite(x)
            return x
        elseif isnan(x)
            return nothing
        elseif x == Inf
            return "Inf"
        elseif x == -Inf
            return "-Inf"
        end
    elseif x isa Dict
        return Dict(string(k) => sanitize_json(v) for (k,v) in x)
    elseif x isa AbstractVector
        return [sanitize_json(v) for v in x]
    else
        return x
    end
end

function save_json(path::String, obj)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, sanitize_json(obj), 2)
    end
    println("Saved JSON: ", path)
end

function main()
    coarse_n1 = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : COARSE_N1_DEFAULT
    coarse_n2 = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : COARSE_N2_DEFAULT
    fine_n1 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : FINE_N1_DEFAULT
    fine_n2 = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : FINE_N2_DEFAULT
    wf_n1 = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : WF_N1_DEFAULT
    wf_n2 = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : WF_N2_DEFAULT
    x0_n1 = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : X0_N1_DEFAULT
    x0_n2 = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : X0_N2_DEFAULT
    coeff_bound = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 1.0e4

    result = solve_legacy_grid(; coarse_n1=coarse_n1, coarse_n2=coarse_n2, fine_n1=fine_n1, fine_n2=fine_n2, wf_n1=wf_n1, wf_n2=wf_n2, x0_n1=x0_n1, x0_n2=x0_n2, coeff_bound=coeff_bound)
    out_path = joinpath("results", "pccc_synth_platoon_legacy_pwquad_grid.json")
    save_json(out_path, result)
end

main()
