# Author: Reza Iraji
# PC-CC / SOS synthesis for the two-car platoon example.
#
# This version implements the implication-style pairwise PC-CC conditions:
#
# P1. One-step closure:
#     For each edge (p, sigma, q):
#         C[p,q](x, f_sigma(x)) >= 0.
#
# P2. Transitive-closure implication:
#     For each edge (p, sigma, q) and each node r:
#         C[q,r](f_sigma(x), y) >= 0  ==>  C[p,r](x,y) >= 0.
#
# P3. Well-foundedness / ranking decrease on Xu:
#     For all p,q,r:
#         C[p,q](x0,y) >= 0 and C[q,r](y,z) >= 0
#             ==> C[p,r](x0,z) <= C[p,q](x0,y) - WF_DEC.
#
# Run from the repository root with:
#
#     julia --project=. sos/pccc_synth_platoon_sos.jl
#
# Main outputs if feasible:
#
#     results/pccc_synth_platoon_sos.jld2
#     results/pccc_synth_platoon_sos.json
#
# If infeasible:
#
#     results/pccc_synth_platoon_sos_status.json

using JuMP
using SumOfSquares
using DynamicPolynomials
using MultivariatePolynomials
using MosekTools
using SCS
using JLD2
using JSON

const MOI = JuMP.MOI

# User parameters
# ---------------------------------------------------------------------

# Strict ranking decrease in the well-foundedness implication.
const WF_DEC = 1.0e-6

# --------------------------------------------------------------------------
# Implication multiplier configuration
#
# The closure certificate C[p,q](x,y) remains quadratic. These settings only
# control the S-procedure multipliers used inside implication SOS constraints.
#
# USE_SOS_PREMISE_MULTIPLIERS = false:
#     Use the old fixed scalar multipliers:
#
#         conclusion - IMP_MULT_P2 * premise
#
#     and:
#
#         conclusion - IMP_MULT_P3_A * premise1
#                    - IMP_MULT_P3_B * premise2
#
# USE_SOS_PREMISE_MULTIPLIERS = true:
#     Replace the fixed constants by SOS polynomial multipliers.
#
# Important:
#     If C and the SOS premise multipliers are both decision variables, the
#     resulting model is bilinear. Therefore this mode should ultimately be
#     used inside an alternating fixed-C / fixed-multiplier scheme.
# --------------------------------------------------------------------------

const USE_SOS_PREMISE_MULTIPLIERS = false

# Old fixed scalar multipliers. These are kept as a baseline and fallback.
const IMP_MULT_P2   = 0.001
const IMP_MULT_P3_A = 0.01
const IMP_MULT_P3_B = 0.01

# Degree of SOS premise multipliers.
#
# Degree 0:
#     s(x) is just a nonnegative scalar decision variable.
#
# Degree 2:
#     s(x) is a quadratic SOS polynomial.
#
# Start with degree 0. Only move to degree 2 after the scalar-SOS version
# is implemented and tested.
const P2_PREMISE_MULT_DEG = 0
const P3_PREMISE_MULT_DEG = 0

const IMP_MULTIPLIER_MODE =
    USE_SOS_PREMISE_MULTIPLIERS ? "sos_premise_multipliers" : "fixed_scalar_multipliers"

const RESULTS_DIR = "results"
const OUT_JLD2 = joinpath(RESULTS_DIR, "pccc_synth_platoon_sos.jld2")
const OUT_JSON = joinpath(RESULTS_DIR, "pccc_synth_platoon_sos.json")
const OUT_STATUS_JSON = joinpath(RESULTS_DIR, "pccc_synth_platoon_sos_status.json")

# Constant nonnegative multipliers are used in this first SOS version.
# This gives a conservative S-procedure certificate.
#
# If the problem is infeasible, the next upgrade is to replace the constant
# multipliers by polynomial SOS multipliers, or to use an alternating scheme.
const USE_SILENT_SOLVER = false

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

# Platoon dynamics
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

println("Implication multiplier mode = ", IMP_MULTIPLIER_MODE)

if USE_SOS_PREMISE_MULTIPLIERS
    println("P2 premise multiplier degree = ", P2_PREMISE_MULT_DEG)
    println("P3 premise multiplier degree = ", P3_PREMISE_MULT_DEG)
else
    println("Fixed implication multipliers:")
    println("  IMP_MULT_P2   = ", IMP_MULT_P2)
    println("  IMP_MULT_P3_A = ", IMP_MULT_P3_A)
    println("  IMP_MULT_P3_B = ", IMP_MULT_P3_B)
end

# Solver constructors
# ---------------------------------------------------------------------

function make_model_with_mosek()
    model = SOSModel(Mosek.Optimizer)

    if USE_SILENT_SOLVER
        set_silent(model)
    end

    return model
end

function make_model_with_scs()
    optimizer = optimizer_with_attributes(
        SCS.Optimizer,
        "eps_abs" => 1.0e-8,
        "eps_rel" => 1.0e-8,
        "eps_infeas" => 1.0e-9,
        "max_iters" => 2000000,
        "verbose" => 1,
    )

    model = SOSModel(optimizer)

    if USE_SILENT_SOLVER
        set_silent(model)
    end

    return model
end

# Polynomial substitution helpers
# ---------------------------------------------------------------------

function C_eval(Cpq, a1, a2, b1, b2)
    return subs(
        Cpq,
        x1 => a1,
        x2 => a2,
        y1 => b1,
        y2 => b2,
    )
end

function C_x_fx(Cpq, sigma)
    fx1, fx2 = f_sigma(sigma, x1, x2)
    return C_eval(Cpq, x1, x2, fx1, fx2)
end

function C_fx_y(Cpq, sigma)
    fx1, fx2 = f_sigma(sigma, x1, x2)
    return C_eval(Cpq, fx1, fx2, y1, y2)
end

function C_x_y(Cpq)
    return C_eval(Cpq, x1, x2, y1, y2)
end

function C_x0_y(Cpq)
    return C_eval(Cpq, x01, x02, y1, y2)
end

function C_x0_z(Cpq)
    return C_eval(Cpq, x01, x02, z1, z2)
end

function C_y_z(Cpq)
    return C_eval(Cpq, y1, y2, z1, z2)
end

# SOS/S-procedure helper constraints
# ---------------------------------------------------------------------

function add_domain_sos_constraint!(model, poly_nonnegative, domain_cons; label="")
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

function new_sos_multiplier!(model, vars, degree)
    @assert degree >= 0
    @assert iseven(degree) "SOS multiplier degree should be even, e.g. 0 or 2."

    mult_basis = monomials(vars, 0:degree)

    # Anonymous JuMP coefficient vector.
    # Do not use @variable(model, c[...]) here, because repeated calls would
    # try to register the same JuMP name c in the model.
    coeffs = @variable(
        model,
        [1:length(mult_basis)],
        base_name = string(gensym(:sosmult))
    )

    s = sum(coeffs[i] * mult_basis[i] for i in 1:length(mult_basis))

    @constraint(model, s in SOSCone())

    return s
end

function add_implication_sos_constraint!(
    model,
    conclusion_poly,
    premise_poly,
    domain_cons;
    label=""
)
    m = length(domain_cons)

    mu = @variable(model, [1:m], lower_bound = 0.0)

    if USE_SOS_PREMISE_MULTIPLIERS
        vars_p2 = [x1, x2, y1, y2]
        s_premise = new_sos_multiplier!(model, vars_p2, P2_PREMISE_MULT_DEG)
        certificate_poly = conclusion_poly - s_premise * premise_poly
    else
        certificate_poly = conclusion_poly - IMP_MULT_P2 * premise_poly
    end

    for i in 1:m
        certificate_poly -= mu[i] * domain_cons[i]
    end

    @constraint(model, certificate_poly in SOSCone())

    return
end

function add_two_premise_implication_sos_constraint!(
    model,
    conclusion_poly,
    premise1_poly,
    premise2_poly,
    domain_cons;
    label=""
)
    m = length(domain_cons)

    mu = @variable(model, [1:m], lower_bound = 0.0)

    if USE_SOS_PREMISE_MULTIPLIERS
        vars_p3 = [x01, x02, y1, y2, z1, z2]

        s_premise1 = new_sos_multiplier!(model, vars_p3, P3_PREMISE_MULT_DEG)
        s_premise2 = new_sos_multiplier!(model, vars_p3, P3_PREMISE_MULT_DEG)

        certificate_poly =
            conclusion_poly -
            s_premise1 * premise1_poly -
            s_premise2 * premise2_poly
    else
        certificate_poly =
            conclusion_poly -
            IMP_MULT_P3_A * premise1_poly -
            IMP_MULT_P3_B * premise2_poly
    end

    for i in 1:m
        certificate_poly -= mu[i] * domain_cons[i]
    end

    @constraint(model, certificate_poly in SOSCone())

    return
end

# Build SOS problem
# ---------------------------------------------------------------------

function build_sos_problem(model)
    @variable(model, coeff[p in nodes, q in nodes, k in 1:nbasis])

    C = Matrix{Any}(undef, length(nodes), length(nodes))

    for p in nodes
        for q in nodes
            C[p, q] = sum(coeff[p, q, k] * basis_xy[k] for k in 1:nbasis)
        end
    end

    num_sos_constraints = 0

    # -----------------------------------------------------------------
    # P1. One-step closure:
    #
    # For each graph edge (p, sigma, q):
    #
    #     C_{p,q}(x, f_sigma(x)) >= 0
    #
    # for every x in X.
    # -----------------------------------------------------------------

    for (p, sigma, q) in edges
        poly = C_x_fx(C[p, q], sigma)

        add_domain_sos_constraint!(
            model,
            poly,
            X_cons_x;
            label = "one_step"
        )

        num_sos_constraints += 1
    end

    # -----------------------------------------------------------------
    # P2. Transitive-closure implication:
    #
    # For each edge (p, sigma, q), each target node r, and all (x,y) in X x X:
    #
    #     C_{q,r}(f_sigma(x), y) >= 0
    #         ==> C_{p,r}(x,y) >= 0.
    # -----------------------------------------------------------------

    XY_cons = vcat(X_cons_x, X_cons_y)

    for (p, sigma, q) in edges
        for r in nodes
            premise = C_fx_y(C[q, r], sigma)
            conclusion = C_x_y(C[p, r])

            add_implication_sos_constraint!(
                model,
                conclusion,
                premise,
                XY_cons;
                label = "transitive"
            )

            num_sos_constraints += 1
        end
    end

    # -----------------------------------------------------------------
    # P3. Well-foundedness / ranking decrease on Xu:
    #
    # For all p,q,r, x0 in X0, and y,z in Xu:
    #
    #     C_{p,q}(x0,y) >= 0 and C_{q,r}(y,z) >= 0
    #         ==> C_{p,r}(x0,z) <= C_{p,q}(x0,y) - WF_DEC.
    #
    # Equivalently:
    #
    #     C_{p,q}(x0,y) - WF_DEC - C_{p,r}(x0,z) >= 0
    #
    # under the two premises.
    # -----------------------------------------------------------------

    X0_Xu_Xu_cons = vcat(X0_cons, Xu_cons_y, Xu_cons_z)

    for p in nodes
        for q in nodes
            for r in nodes
                premise1 = C_x0_y(C[p, q])
                premise2 = C_y_z(C[q, r])

                conclusion =
                    C_x0_y(C[p, q]) -
                    WF_DEC -
                    C_x0_z(C[p, r])

                add_two_premise_implication_sos_constraint!(
                    model,
                    conclusion,
                    premise1,
                    premise2,
                    X0_Xu_Xu_cons;
                    label = "well_founded"
                )

                num_sos_constraints += 1
            end
        end
    end

    # -----------------------------------------------------------------
    # Pure feasibility objective.
    # -----------------------------------------------------------------

    @objective(model, Min, 0.0)

    return coeff, C, num_sos_constraints
end

# Solve, validate, extract, save
# ---------------------------------------------------------------------

function solve_with_solver(solver_name::Symbol)
    if solver_name == :mosek
        println("\nBuilding implication-style SOS model with MOSEK...")
        model = make_model_with_mosek()
    elseif solver_name == :scs
        println("\nBuilding implication-style SOS model with SCS fallback...")
        model = make_model_with_scs()
    else
        error("Unknown solver_name = $solver_name")
    end

    coeff, C, num_sos_constraints = build_sos_problem(model)

    println("Number of SOS/domain constraints = ", num_sos_constraints)
    println("Expected count for full implication model = 20")
    println("Starting optimization...")

    optimize!(model)

    status = termination_status(model)
    pstatus = primal_status(model)

    println("\nSolver = ", solver_name)
    println("Termination status = ", status)
    println("Primal status      = ", pstatus)

    if has_values(model)
        try
            obj = objective_value(model)
            println("Objective value    = ", obj)
        catch err
            println("Objective value unavailable: ", err)
        end
    else
        println("Objective value    = unavailable because no primal values were returned.")
    end

    return model, coeff, C
end

function is_successful_certificate(model)
    term = termination_status(model)
    prim = primal_status(model)

    good_term = term in [
        MOI.OPTIMAL,
        MOI.ALMOST_OPTIMAL,
        MOI.LOCALLY_SOLVED,
        MOI.ALMOST_LOCALLY_SOLVED,
    ]

    good_primal = prim in [
        MOI.FEASIBLE_POINT,
        MOI.NEARLY_FEASIBLE_POINT,
    ]

    return good_term && good_primal && has_values(model)
end

function safe_objective_value(model)
    if !has_values(model)
        return nothing
    end

    obj = try
        objective_value(model)
    catch
        NaN
    end

    if obj isa Number && isfinite(obj)
        return obj
    else
        return nothing
    end
end

function extract_coefficients(model, coeff)
    coeff_values = zeros(length(nodes), length(nodes), nbasis)

    for p in nodes
        for q in nodes
            for k in 1:nbasis
                val = value(coeff[p, q, k])

                if !(val isa Number) || !isfinite(val)
                    error(
                        "Invalid coefficient value at (p,q,k)=($p,$q,$k): $val. " *
                        "This usually means the solver did not return a feasible certificate."
                    )
                end

                coeff_values[p, q, k] = val
            end
        end
    end

    return coeff_values
end

function save_status_only(model, solver_name::Symbol)
    mkpath(RESULTS_DIR)

    status_string = string(termination_status(model))
    primal_status_string = string(primal_status(model))
    objective = safe_objective_value(model)

    edge_array = [[p, sigma, q] for (p, sigma, q) in edges]

    json_dict = Dict(
        "description" => "Implication-style SOS PC-CC synthesis status for two-car platoon example",
        "certificate_type" => "implication_pairwise_sos",
        "solver" => string(solver_name),
        "status" => status_string,
        "primal_status" => primal_status_string,
        "objective" => objective,
        "feasible_certificate_saved" => false,
        "message" => "No coefficient file was saved because the solver did not return a feasible certificate.",
        "wf_decrease" => WF_DEC,
        "implication_multipliers" => Dict(
            "IMP_MULT_P2" => IMP_MULT_P2,
            "IMP_MULT_P3_A" => IMP_MULT_P3_A,
            "IMP_MULT_P3_B" => IMP_MULT_P3_B
        ),
        "nodes" => nodes,
        "edges" => edge_array,
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

    open(OUT_STATUS_JSON, "w") do io
        JSON.print(io, json_dict, 2)
    end

    println("\nSaved infeasibility/status summary:")
    println("  ", OUT_STATUS_JSON)
end

function save_results(model, coeff_values, solver_name::Symbol)
    mkpath(RESULTS_DIR)

    basis_strings = [string(b) for b in basis_xy]
    edge_array = [[p, sigma, q] for (p, sigma, q) in edges]

    status_string = string(termination_status(model))
    primal_status_string = string(primal_status(model))
    objective = safe_objective_value(model)

    certificate_type = "implication_pairwise_sos"

    @save OUT_JLD2 coeff_values basis_strings edge_array WF_DEC status_string primal_status_string objective certificate_type

    json_dict = Dict(
        "description" => "Implication-style SOS PC-CC synthesis result for two-car platoon example",
        "certificate_type" => certificate_type,
        "solver" => string(solver_name),
        "status" => status_string,
        "primal_status" => primal_status_string,
        "objective" => objective,
        "feasible_certificate_saved" => true,
        "wf_decrease" => WF_DEC,
        "implication_multipliers" => Dict(
            "IMP_MULT_P2" => IMP_MULT_P2,
            "IMP_MULT_P3_A" => IMP_MULT_P3_A,
            "IMP_MULT_P3_B" => IMP_MULT_P3_B
        ),
        "nodes" => nodes,
        "edges" => edge_array,
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
        ),
        "conditions" => Dict(
            "P1" => "For each edge (p,sigma,q), C[p,q](x,f_sigma(x)) >= 0 on X.",
            "P2" => "For each edge (p,sigma,q) and node r, C[q,r](f_sigma(x),y) >= 0 implies C[p,r](x,y) >= 0 on X x X.",
            "P3" => "For all p,q,r, C[p,q](x0,y) >= 0 and C[q,r](y,z) >= 0 imply C[p,r](x0,z) <= C[p,q](x0,y) - wf_decrease on X0 x Xu x Xu."
        )
    )

    open(OUT_JSON, "w") do io
        JSON.print(io, json_dict, 2)
    end

    println("\nSaved feasible certificate results:")
    println("  ", OUT_JLD2)
    println("  ", OUT_JSON)
end

function main()
    model = nothing
    coeff = nothing
    C = nothing
    solver_used = :none

    # First try MOSEK.
    try
        model, coeff, C = solve_with_solver(:mosek)
        solver_used = :mosek

        if !is_successful_certificate(model)
            println("\nMOSEK did not return a feasible certificate.")
            println("Termination status = ", termination_status(model))
            println("Primal status      = ", primal_status(model))
            println("\nTrying SCS fallback...")

            model, coeff, C = solve_with_solver(:scs)
            solver_used = :scs
        end
    catch err
        println("\nMOSEK solve failed.")
        println("Error:")
        println(err)

        println("\nTrying SCS fallback...")

        try
            model, coeff, C = solve_with_solver(:scs)
            solver_used = :scs
        catch err2
            println("\nSCS fallback also failed.")
            println("Error:")
            println(err2)
            return
        end
    end

    if !is_successful_certificate(model)
        println("\n[WARN] SOS synthesis did not return a feasible certificate.")
        println("       Skipping coefficient extraction and full JSON/JLD2 save.")
        println("       Termination status = ", termination_status(model))
        println("       Primal status      = ", primal_status(model))
        println()
        println("Current implication multiplier mode = ", IMP_MULTIPLIER_MODE)
        println("Current fixed implication multipliers:")
        println("  IMP_MULT_P2   = ", IMP_MULT_P2)
        println("  IMP_MULT_P3_A = ", IMP_MULT_P3_A)
        println("  IMP_MULT_P3_B = ", IMP_MULT_P3_B)
        println()
        println("Recommended next trials:")
        if USE_SOS_PREMISE_MULTIPLIERS
            println("  1) If this was a direct joint solve, remember that C and SOS premise multipliers make the problem bilinear.")
            println("  2) If the solver errors or behaves poorly, move to an alternating fixed-C / fixed-multiplier implementation.")
            println("  3) Start with P2_PREMISE_MULT_DEG = 0 and P3_PREMISE_MULT_DEG = 0.")
            println("  4) Then try P2_PREMISE_MULT_DEG = 0 and P3_PREMISE_MULT_DEG = 2.")
        else
            println("  1) Step 2/3/4/5 infrastructure is installed and compiles.")
            println("  2) Next test: set USE_SOS_PREMISE_MULTIPLIERS = true.")
            println("  3) Keep P2_PREMISE_MULT_DEG = 0 and P3_PREMISE_MULT_DEG = 0 for the first SOS-multiplier test.")
            println("  4) If the direct model fails because of bilinearity, implement the alternating fixed-C / fixed-multiplier scheme.")
        end

        save_status_only(model, solver_used)

        println("\nDone, but no certificate was saved.")
        return
    end

    println("\nFeasible SOS certificate found. Extracting coefficients...")

    coeff_values = extract_coefficients(model, coeff)
    save_results(model, coeff_values, solver_used)

    println("\nDone.")
end

main()