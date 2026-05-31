# Author: Reza Iraji
# PC-CC / SOS synthesis baseline for the two-car platoon example.
#
# Run from the repository root with:
#
#     julia --project=. sos/pccc_synth_platoon_sos.jl
#
# Main outputs:
#
#     results/pccc_synth_platoon_sos.jld2
#     results/pccc_synth_platoon_sos.json

using JuMP
using SumOfSquares
using DynamicPolynomials
using MultivariatePolynomials
using MosekTools
using SCS
using JLD2
using JSON

# User parameters
# ---------------------------------------------------------------------

const ZETA    = 3.0
const LAMBDA1 = 0.95

const LAMBDA2 = 0.02
const LAMBDA3 = 0.02
const THETA   = 5.0e-5

const RESULTS_DIR = "results"
const OUT_JLD2 = joinpath(RESULTS_DIR, "pccc_synth_platoon_sos.jld2")
const OUT_JSON = joinpath(RESULTS_DIR, "pccc_synth_platoon_sos.json")

# Constant nonnegative multipliers are used in this first baseline.
# This gives a conservative Putinar/S-procedure certificate.
#
# If the problem is infeasible, the next upgrade is to replace the constant
# multipliers in add_domain_sos_constraint! with polynomial SOS multipliers.
const USE_SILENT_SOLVER = false

# Only this pair is used for the well-foundedness/ranking condition.
# This mirrors the old Python script, which used one ranking node.
#
# We can make this stronger by changing to:
#
#     const RANK_PAIRS = [(p, q) for p in nodes for q in nodes]
#
# but that may make the SOS problem harder or infeasible.
const RANK_PAIRS = [(1, 1)]

# Polynomial variables
# ---------------------------------------------------------------------

@polyvar x1 x2 y1 y2 z1 z2 x01 x02

const nodes = [1, 2]

# Edge convention:
#
#     (p, sigma, q)
#
# means mode sigma maps source node p to target node q.
const edges = [
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 2),
    (2, 2, 1),
]

# Dynamics
# ---------------------------------------------------------------------

f1_1(a1, a2) = 0.01 * a2 + 0.9 * a1 - 0.02 * a1^2
f1_2(a1, a2) = 2.0 + 0.8 * a2 - 0.04 * a2^2

f2_1(a1, a2) = 0.9 * a1 - 0.02 * a1^2
f2_2(a1, a2) = 2.0 + 0.8 * a2 - 0.04 * a2^2

function f_sigma(sigma, a1, a2)
    if sigma == 1
        return f1_1(a1, a2), f1_2(a1, a2)
    elseif sigma == 2
        return f2_1(a1, a2), f2_2(a1, a2)
    else
        error("Unknown mode sigma = $sigma. Expected 1 or 2.")
    end
end

# Semialgebraic domain polynomials
# ---------------------------------------------------------------------
#
# A constraint list [g1, ..., gm] means:
#
#     gi >= 0
#
# for every i.
#
# State-space domain X:
#
#     0 <= x1 <= 3
#     0 <= x2 <= 5.1
#     x2 - x1 >= 0
#
# Initial set X0:
#
#     X plus x02 - x01 <= 0.3
#
# Unsafe set Xu:
#
#     X plus x2 - x1 <= 0.4

X_cons_x = [
    x1,
    3.0 - x1,
    x2,
    5.1 - x2,
    x2 - x1,
]

X_cons_y = [
    y1,
    3.0 - y1,
    y2,
    5.1 - y2,
    y2 - y1,
]

X0_cons = [
    x01,
    3.0 - x01,
    x02,
    5.1 - x02,
    x02 - x01,
    0.3 - (x02 - x01),
]

Xu_cons_y = [
    y1,
    3.0 - y1,
    y2,
    5.1 - y2,
    y2 - y1,
    0.4 - (y2 - y1),
]

Xu_cons_z = [
    z1,
    3.0 - z1,
    z2,
    5.1 - z2,
    z2 - z1,
    0.4 - (z2 - z1),
]

# Polynomial template C_{p,q}(x,y)
# ---------------------------------------------------------------------

basis_xy = monomials([x1, x2, y1, y2], 0:2)
nbasis = length(basis_xy)

println("Using quadratic template for C[p,q](x,y).")
println("Number of basis monomials = ", nbasis)

# Helper functions
# ---------------------------------------------------------------------

function make_model_with_mosek()
    model = SOSModel(Mosek.Optimizer)
    if USE_SILENT_SOLVER
        set_silent(model)
    end
    return model
end

function make_model_with_scs()
    # Optional fallback. This requires SCS to be present in Project.toml.
    model = SOSModel(SCS.Optimizer)

    if USE_SILENT_SOLVER
        set_silent(model)
    end

    return model
end

function C_eval(Cpq, a1, a2, b1, b2)
    # Evaluate Cpq(x,y) at:
    #
    #     x = (a1, a2)
    #     y = (b1, b2)
    #
    # The arguments may themselves be polynomial expressions.
    return subs(
        Cpq,
        x1 => a1,
        x2 => a2,
        y1 => b1,
        y2 => b2,
    )
end

function C_x_fx(Cpq, sigma)
    # C_{p,q}(x, f_sigma(x))
    fx1, fx2 = f_sigma(sigma, x1, x2)
    return C_eval(Cpq, x1, x2, fx1, fx2)
end

function C_fx_y(Cpq, sigma)
    # C_{q,r}(f_sigma(x), y)
    fx1, fx2 = f_sigma(sigma, x1, x2)
    return C_eval(Cpq, fx1, fx2, y1, y2)
end

function C_x_y(Cpq)
    # C_{p,r}(x,y)
    return C_eval(Cpq, x1, x2, y1, y2)
end

function C_x0_z(Cpq)
    # C(x0,z)
    return C_eval(Cpq, x01, x02, z1, z2)
end

function C_y_x0(Cpq)
    # C(y,x0)
    return C_eval(Cpq, y1, y2, x01, x02)
end

function C_x0_y(Cpq)
    # C(x0,y)
    return C_eval(Cpq, x01, x02, y1, y2)
end

function C_y_z(Cpq)
    # C(y,z)
    return C_eval(Cpq, y1, y2, z1, z2)
end

function add_domain_sos_constraint!(model, poly_nonnegative, domain_cons; label="")
    # Enforce:
    #
    #     poly_nonnegative >= 0
    #
    # on the semialgebraic set:
    #
    #     domain_cons[i] >= 0.
    #
    # This baseline uses constant nonnegative multipliers mu_i >= 0 and enforces:
    #
    #     poly_nonnegative - sum_i mu_i * domain_cons[i] is SOS.
    #
    # This is conservative but simple and robust.

    m = length(domain_cons)

    if m == 0
        @constraint(model, poly_nonnegative in SOSCone())
        return
    end

    mu = @variable(model, [1:m], lower_bound = 0.0)

    certificate_poly = poly_nonnegative
    for i in 1:m
        certificate_poly -= mu[i] * domain_cons[i]
    end

    @constraint(model, certificate_poly in SOSCone())

    return
end

function build_sos_problem(model)
    # Decision variables for the polynomial coefficients of C[p,q](x,y).
    @variable(model, coeff[p in nodes, q in nodes, k in 1:nbasis])

    C = Matrix{Any}(undef, length(nodes), length(nodes))

    for p in nodes
        for q in nodes
            C[p, q] = sum(coeff[p, q, k] * basis_xy[k] for k in 1:nbasis)
        end
    end

    num_sos_constraints = 0

    # -----------------------------------------------------------------
    # Constraint family A:
    #
    # Transition boundedness:
    #
    #     C_{p,q}(x, f_sigma(x)) <= zeta
    #
    # for every graph edge (p, sigma, q) and every x in X.
    #
    # SOS form:
    #
    #     zeta - C_{p,q}(x, f_sigma(x)) >= 0 on X.
    # -----------------------------------------------------------------

    for (p, sigma, q) in edges
        poly = ZETA - C_x_fx(C[p, q], sigma)
        add_domain_sos_constraint!(model, poly, X_cons_x; label="transition")
        num_sos_constraints += 1
    end

    # -----------------------------------------------------------------
    # Constraint family B:
    #
    # Forward path-complete closure:
    #
    #     C_{p,r}(x,y)
    #         <= lambda1 C_{q,r}(f_sigma(x),y) + (1-lambda1) zeta
    #
    # for every edge (p, sigma, q), every target r, and all (x,y) in X x X.
    #
    # SOS form:
    #
    #     lambda1 C_{q,r}(f_sigma(x),y)
    #       + (1-lambda1) zeta
    #       - C_{p,r}(x,y)
    #       >= 0
    #
    # on X x X.
    # -----------------------------------------------------------------

    XY_cons = vcat(X_cons_x, X_cons_y)

    for (p, sigma, q) in edges
        for r in nodes
            poly =
                LAMBDA1 * C_fx_y(C[q, r], sigma) +
                (1.0 - LAMBDA1) * ZETA -
                C_x_y(C[p, r])

            add_domain_sos_constraint!(model, poly, XY_cons; label="forward")
            num_sos_constraints += 1
        end
    end

    # -----------------------------------------------------------------
    # Constraint family C:
    #
    # Well-foundedness / finite visits of Xu:
    #
    # For x0 in X0 and y,z in Xu:
    #
    #     C(x0,z) + C(y,x0)
    #         <= lambda2 C(x0,y)
    #            + lambda3 C(y,z)
    #            - (theta + lambda2*zeta + lambda3*zeta)
    #
    # SOS form:
    #
    #     lambda2 C(x0,y)
    #       + lambda3 C(y,z)
    #       - C(x0,z)
    #       - C(y,x0)
    #       - (theta + lambda2*zeta + lambda3*zeta)
    #       >= 0
    #
    # on X0 x Xu x Xu.
    #
    # By default this is imposed only for RANK_PAIRS = [(1,1)].
    # -----------------------------------------------------------------

    X0_Xu_Xu_cons = vcat(X0_cons, Xu_cons_y, Xu_cons_z)

    for (p, q) in RANK_PAIRS
        poly =
            LAMBDA2 * C_x0_y(C[p, q]) +
            LAMBDA3 * C_y_z(C[p, q]) -
            C_x0_z(C[p, q]) -
            C_y_x0(C[p, q]) -
            (THETA + LAMBDA2 * ZETA + LAMBDA3 * ZETA)

        add_domain_sos_constraint!(model, poly, X0_Xu_Xu_cons; label="well_founded")
        num_sos_constraints += 1
    end

    # -----------------------------------------------------------------
    # Regularization objective.
    #
    # This selects a small-norm certificate among feasible certificates.
    # -----------------------------------------------------------------

    @objective(
        model,
        Min,
        sum(coeff[p, q, k]^2 for p in nodes for q in nodes for k in 1:nbasis)
    )

    return coeff, C, num_sos_constraints
end

function solve_with_solver(solver_name::Symbol)
    if solver_name == :mosek
        println("\nBuilding SOS model with MOSEK...")
        model = make_model_with_mosek()
    elseif solver_name == :scs
        println("\nBuilding SOS model with SCS fallback...")
        model = make_model_with_scs()
    else
        error("Unknown solver_name = $solver_name")
    end

    coeff, C, num_sos_constraints = build_sos_problem(model)

    println("Number of SOS/domain constraints = ", num_sos_constraints)
    println("Starting optimization...")

    optimize!(model)

    status = termination_status(model)
    pstatus = primal_status(model)

    println("\nSolver = ", solver_name)
    println("Termination status = ", status)
    println("Primal status      = ", pstatus)

    if has_values(model)
        try
            println("Objective value    = ", objective_value(model))
        catch err
            println("Objective value unavailable: ", err)
        end
    end

    return model, coeff, C
end

function extract_coefficients(model, coeff)
    coeff_values = zeros(length(nodes), length(nodes), nbasis)

    for p in nodes
        for q in nodes
            for k in 1:nbasis
                coeff_values[p, q, k] = value(coeff[p, q, k])
            end
        end
    end

    return coeff_values
end

function save_results(model, coeff_values)
    mkpath(RESULTS_DIR)

    basis_strings = [string(b) for b in basis_xy]
    edge_array = [[p, sigma, q] for (p, sigma, q) in edges]
    rank_pair_array = [[p, q] for (p, q) in RANK_PAIRS]

    status_string = string(termination_status(model))
    primal_status_string = string(primal_status(model))

    objective = try
        objective_value(model)
    catch
        NaN
    end

    @save OUT_JLD2 coeff_values basis_strings edge_array rank_pair_array ZETA LAMBDA1 LAMBDA2 LAMBDA3 THETA status_string primal_status_string objective

    json_dict = Dict(
        "description" => "SOS PC-CC synthesis result for two-car platoon example",
        "status" => status_string,
        "primal_status" => primal_status_string,
        "objective" => objective,
        "zeta" => ZETA,
        "lambda1" => LAMBDA1,
        "lambda2" => LAMBDA2,
        "lambda3" => LAMBDA3,
        "theta" => THETA,
        "nodes" => nodes,
        "edges" => edge_array,
        "rank_pairs" => rank_pair_array,
        "basis_xy" => basis_strings,
        "coeff_values" => coeff_values,
        "state_domain" => Dict(
            "x1_min" => 0.0,
            "x1_max" => 3.0,
            "x2_min" => 0.0,
            "x2_max" => 5.1,
            "gap_min" => 0.0
        ),
        "initial_set" => Dict(
            "gap_max" => 0.3
        ),
        "unsafe_set" => Dict(
            "gap_max" => 0.4
        )
    )

    open(OUT_JSON, "w") do io
        JSON.print(io, json_dict, 2)
    end

    println("\nSaved results:")
    println("  ", OUT_JLD2)
    println("  ", OUT_JSON)
end

function main()
    model = nothing
    coeff = nothing
    C = nothing

    # First try MOSEK.
    try
        model, coeff, C = solve_with_solver(:mosek)
    catch err
        println("\nMOSEK solve failed.")
        println("Error:")
        println(err)

        println("\nTrying SCS fallback...")
        try
            model, coeff, C = solve_with_solver(:scs)
        catch err2
            println("\nSCS fallback also failed.")
            println("Error:")
            println(err2)
            return
        end
    end

    if !has_values(model)
        println("\nNo primal values were returned. Nothing to save.")
        println("Termination status = ", termination_status(model))
        println("Primal status      = ", primal_status(model))
        return
    end

    coeff_values = extract_coefficients(model, coeff)
    save_results(model, coeff_values)

    println("\nDone.")
end

main()