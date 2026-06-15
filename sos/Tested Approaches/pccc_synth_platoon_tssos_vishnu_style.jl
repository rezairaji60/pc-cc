# Author: Reza Iraji
# PC-CC / TSSOS synthesis for the two-car platoon example.
#
# This is a first TSSOS port of the implication-style pairwise PC-CC search.
# It intentionally starts with the convex fixed-scalar implication multipliers
# version, not the alternating bilinear multiplier version.
#
# It uses TSSOS.add_psatz! to impose Putinar-style SOS nonnegativity
# certificates over the semialgebraic domains.
#
# Run from the repository root with:
#
#   julia --project=. sos/pccc_synth_platoon_tssos.jl
#
# Outputs if a solver-accepted certificate is found:
#
#   results/pccc_synth_platoon_tssos.jld2
#   results/pccc_synth_platoon_tssos.json
#
# The existing validator can then be run with:
#
#   julia --project=. sos/pccc_validate_platoon_sos.jl results/pccc_synth_platoon_tssos.json 21 41

using JuMP
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using TSSOS
using JLD2
using JSON

const MOI = JuMP.MOI

# ---------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------

# Strict ranking decrease in the well-foundedness implication.
const WF_DEC = 1.0e-6

# Quadratic closure-certificate template C[p,q](x,y).
const C_DEGREE = 2

# Fixed scalar S-procedure multipliers for implication premises.
# This keeps the first TSSOS implementation convex.
const IMP_MULT_P2   = 0.001
const IMP_MULT_P3_A = 0.01
const IMP_MULT_P3_B = 0.01

# TSSOS / Putinar relaxation orders.
# P1 and P2 contain C(x,f(x)), which has degree up to 4, hence order 2.
# P3 is degree 2, but order 2 gives extra Putinar multiplier flexibility.
const SOS_TOL = 8

# Vishnu-style Putinar order: div(C_DEGREE + SOS_TOL, 2) = 5 for C_DEGREE=2.
# Start with P3 order 3 because the 6-variable/18-inequality P3 constraints become large at order 5.
const TSSOS_ORDER_P1 = div(C_DEGREE + SOS_TOL, 2)
const TSSOS_ORDER_P2 = div(C_DEGREE + SOS_TOL, 2)
const TSSOS_ORDER_P3 = 3

# TSSOS sparsity settings.
# CS="MF" enables correlative-sparsity chordal extension.
# TS="block" enables block term-sparsity processing.
const TSSOS_CS = false
const TSSOS_TS = false
const TSSOS_SO = 1
const TSSOS_USE_GROEBNER = true
const TSSOS_QUIET = false

const RESULTS_DIR = "results"
const OUT_JLD2 = joinpath(RESULTS_DIR, "pccc_synth_platoon_tssos_vishnu_style.jld2")
const OUT_JSON = joinpath(RESULTS_DIR, "pccc_synth_platoon_tssos_vishnu_style.json")
const OUT_STATUS_JSON = joinpath(RESULTS_DIR, "pccc_synth_platoon_tssos_vishnu_style_status.json")

const USE_SILENT_SOLVER = false

# Save coefficient candidates when MOSEK returns usable primal values
# but the solver status is not certificate-grade. These candidates are
# for validation/debugging only and must not be reported as proofs.
const SAVE_DIAGNOSTIC_CANDIDATE = true

const ACCEPT_TERMS = [
    MOI.OPTIMAL,
    MOI.ALMOST_OPTIMAL,
    MOI.LOCALLY_SOLVED,
    MOI.ALMOST_LOCALLY_SOLVED,
]

const ACCEPT_PRIMALS = [
    MOI.FEASIBLE_POINT,
    MOI.NEARLY_FEASIBLE_POINT,
]

# ---------------------------------------------------------------------
# Polynomial variables
# ---------------------------------------------------------------------

@polyvar x1 x2 y1 y2 z1 z2

const nodes = [1, 2]

# Edge convention: (p, sigma, q) means mode sigma maps source node p
# to target node q.
const edges = [
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 2),
    (2, 2, 1),
]

# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Semialgebraic domain polynomials. A list [g1,...,gm] means gi >= 0.
# ---------------------------------------------------------------------

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

# Initial set constraints for the first argument of C in P3.
# In the TSSOS P3 constraints we reuse x1,x2 as the dummy x0 variables.
# This avoids a TSSOS conversion issue with a separate x01,x02 variable block,
# and is mathematically equivalent because P3 has its own local variable list.
X0_cons_x = [
    x1,
    3.0 - x1,
    x2,
    5.1 - x2,
    x2 - x1,
    0.3 - (x2 - x1),
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

# ---------------------------------------------------------------------
# Quadratic template basis for C[p,q](x,y)
# ---------------------------------------------------------------------

basis_xy = monomials([x1, x2, y1, y2], 0:C_DEGREE)
nbasis = length(basis_xy)

println("Using TSSOS synthesis.")
println("Using quadratic template for C[p,q](x,y).")
println("Number of basis monomials = ", nbasis)
println("TSSOS orders: P1=", TSSOS_ORDER_P1, ", P2=", TSSOS_ORDER_P2, ", P3=", TSSOS_ORDER_P3)
println("TSSOS settings: CS=", TSSOS_CS, ", TS=", TSSOS_TS, ", SO=", TSSOS_SO, ", GroebnerBasis=", TSSOS_USE_GROEBNER, ", SOS_TOL=", SOS_TOL)
println("Fixed implication multipliers:")
println("  IMP_MULT_P2   = ", IMP_MULT_P2)
println("  IMP_MULT_P3_A = ", IMP_MULT_P3_A)
println("  IMP_MULT_P3_B = ", IMP_MULT_P3_B)

# ---------------------------------------------------------------------
# Solver constructor
# ---------------------------------------------------------------------

function make_model_with_mosek()
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    if USE_SILENT_SOLVER
        set_silent(model)
    end
    return model
end

# ---------------------------------------------------------------------
# Polynomial substitution helpers
# ---------------------------------------------------------------------

function C_eval(Cpq, a1, a2, b1, b2)
    return subs(Cpq, x1 => a1, x2 => a2, y1 => b1, y2 => b2)
end

function C_x_fx(Cpq, sigma)
    fx1, fx2 = f_sigma(sigma, x1, x2)
    return C_eval(Cpq, x1, x2, fx1, fx2)
end

function C_fx_y(Cpq, sigma)
    fx1, fx2 = f_sigma(sigma, x1, x2)
    return C_eval(Cpq, fx1, fx2, y1, y2)
end

C_x_y(Cpq) = C_eval(Cpq, x1, x2, y1, y2)
C_x0_y(Cpq) = C_eval(Cpq, x1, x2, y1, y2)
C_x0_z(Cpq) = C_eval(Cpq, x1, x2, z1, z2)
C_y_z(Cpq) = C_eval(Cpq, y1, y2, z1, z2)

# ---------------------------------------------------------------------
# TSSOS helper
# ---------------------------------------------------------------------

function add_putinar_nonneg!(model, nonneg, vars, ineq_cons, order; label="")
    println("Adding TSSOS Putinar constraint ", label,
            " with order=", order,
            ", nvars=", length(vars),
            ", nineq=", length(ineq_cons),
            ", degree=", maxdegree(nonneg))

    info = add_psatz!(
        model,
        nonneg,
        vars,
        ineq_cons,
        [],
        order;
        CS = TSSOS_CS,
        TS = TSSOS_TS,
        SO = TSSOS_SO,
        QUIET = TSSOS_QUIET,
        GroebnerBasis = TSSOS_USE_GROEBNER,
    )

    return info
end

# ---------------------------------------------------------------------
# Build TSSOS problem
# ---------------------------------------------------------------------

function build_tssos_problem(model)
    @variable(model, coeff[p in nodes, q in nodes, k in 1:nbasis])

    C = Matrix{Any}(undef, length(nodes), length(nodes))

    for p in nodes
        for q in nodes
            C[p, q] = sum(coeff[p, q, k] * basis_xy[k] for k in 1:nbasis)
        end
    end

    constraint_infos = Any[]
    num_psatz_constraints = 0

    # P1. One-step closure:
    #   C[p,q](x, f_sigma(x)) >= 0 on X.
    for (p, sigma, q) in edges
        poly = C_x_fx(C[p, q], sigma)
        info = add_putinar_nonneg!(
            model,
            poly,
            [x1, x2],
            X_cons_x,
            TSSOS_ORDER_P1;
            label = "P1_edge_$(p)_$(sigma)_$(q)",
        )
        push!(constraint_infos, info)
        num_psatz_constraints += 1
    end

    # P2. Transitive implication:
    #   C[q,r](f_sigma(x), y) >= 0 ==> C[p,r](x,y) >= 0 on X x X.
    # Fixed scalar S-procedure:
    #   C[p,r](x,y) - IMP_MULT_P2*C[q,r](f_sigma(x),y) >= 0 on X x X.
    XY_cons = vcat(X_cons_x, X_cons_y)

    for (p, sigma, q) in edges
        for r in nodes
            premise = C_fx_y(C[q, r], sigma)
            conclusion = C_x_y(C[p, r])
            poly = conclusion - IMP_MULT_P2 * premise

            info = add_putinar_nonneg!(
                model,
                poly,
                [x1, x2, y1, y2],
                XY_cons,
                TSSOS_ORDER_P2;
                label = "P2_edge_$(p)_$(sigma)_$(q)_r_$(r)",
            )
            push!(constraint_infos, info)
            num_psatz_constraints += 1
        end
    end

    # P3. Well-foundedness/ranking decrease:
    #   C[p,q](x0,y) >= 0 and C[q,r](y,z) >= 0
    #       ==> C[p,r](x0,z) <= C[p,q](x0,y) - WF_DEC.
    # Fixed scalar S-procedure:
    #   C[p,q](x0,y) - WF_DEC - C[p,r](x0,z)
    #       - IMP_MULT_P3_A*C[p,q](x0,y)
    #       - IMP_MULT_P3_B*C[q,r](y,z) >= 0
    # over X0 x Xu x Xu.
    X0_Xu_Xu_cons = vcat(X0_cons_x, Xu_cons_y, Xu_cons_z)

    for p in nodes
        for q in nodes
            for r in nodes
                premise1 = C_x0_y(C[p, q])
                premise2 = C_y_z(C[q, r])
                conclusion = C_x0_y(C[p, q]) - WF_DEC - C_x0_z(C[p, r])
                poly = conclusion - IMP_MULT_P3_A * premise1 - IMP_MULT_P3_B * premise2

                info = add_putinar_nonneg!(
                    model,
                    poly,
                    [x1, x2, y1, y2, z1, z2],
                    X0_Xu_Xu_cons,
                    TSSOS_ORDER_P3;
                    label = "P3_$(p)_$(q)_$(r)",
                )
                push!(constraint_infos, info)
                num_psatz_constraints += 1
            end
        end
    end

    @objective(model, Min, 0.0)

    return coeff, C, constraint_infos, num_psatz_constraints
end

# ---------------------------------------------------------------------
# Solve, extract, save
# ---------------------------------------------------------------------

function acceptable_solution(model)
    return termination_status(model) in ACCEPT_TERMS &&
           primal_status(model) in ACCEPT_PRIMALS &&
           has_values(model)
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

function extract_coefficients(coeff)
    coeff_values = zeros(length(nodes), length(nodes), nbasis)

    for p in nodes
        for q in nodes
            for k in 1:nbasis
                val = value(coeff[p, q, k])
                if !(val isa Number) || !isfinite(val)
                    error("Invalid coefficient value at (p,q,k)=($p,$q,$k): $val")
                end
                coeff_values[p, q, k] = Float64(val)
            end
        end
    end

    return coeff_values
end

function save_status_only(model, num_psatz_constraints)
    mkpath(RESULTS_DIR)

    status_string = string(termination_status(model))
    primal_status_string = string(primal_status(model))
    objective = safe_objective_value(model)
    edge_array = [[p, sigma, q] for (p, sigma, q) in edges]

    json_dict = Dict(
        "description" => "TSSOS PC-CC synthesis status for two-car platoon example",
        "certificate_type" => "tssos_implication_pairwise_fixed_scalar_vishnu_style",
        "status" => status_string,
        "primal_status" => primal_status_string,
        "objective" => objective,
        "feasible_certificate_saved" => false,
        "message" => "No coefficient file was saved because the solver did not return an accepted certificate.",
        "num_psatz_constraints" => num_psatz_constraints,
        "wf_decrease" => WF_DEC,
        "C_degree" => C_DEGREE,
        "tssos_orders" => Dict(
            "P1" => TSSOS_ORDER_P1,
            "P2" => TSSOS_ORDER_P2,
            "P3" => TSSOS_ORDER_P3,
        ),
        "tssos_sparsity" => Dict(
            "CS" => TSSOS_CS,
            "TS" => TSSOS_TS,
            "SO" => TSSOS_SO,
            "GroebnerBasis" => TSSOS_USE_GROEBNER,
        ),
        "implication_multipliers" => Dict(
            "IMP_MULT_P2" => IMP_MULT_P2,
            "IMP_MULT_P3_A" => IMP_MULT_P3_A,
            "IMP_MULT_P3_B" => IMP_MULT_P3_B,
        ),
        "nodes" => nodes,
        "edges" => edge_array,
    )

    open(OUT_STATUS_JSON, "w") do io
        JSON.print(io, json_dict, 2)
    end

    println("\nSaved TSSOS status summary:")
    println("  ", OUT_STATUS_JSON)
end

function save_results(model, coeff_values, num_psatz_constraints; candidate_only=false)
    mkpath(RESULTS_DIR)

    basis_strings = [string(b) for b in basis_xy]
    edge_array = [[p, sigma, q] for (p, sigma, q) in edges]

    status_string = string(termination_status(model))
    primal_status_string = string(primal_status(model))
    objective = safe_objective_value(model)
    certificate_type = "tssos_implication_pairwise_fixed_scalar_vishnu_style"
    final_solver_accepted = acceptable_solution(model)
    feasible_certificate_saved = final_solver_accepted && !candidate_only

    @save OUT_JLD2 coeff_values basis_strings edge_array WF_DEC status_string primal_status_string objective certificate_type num_psatz_constraints candidate_only feasible_certificate_saved

    json_dict = Dict(
        "description" => "TSSOS implication-style PC-CC synthesis result for two-car platoon example",
        "certificate_type" => certificate_type,
        "status" => status_string,
        "primal_status" => primal_status_string,
        "objective" => objective,
        "feasible_certificate_saved" => feasible_certificate_saved,
        "candidate_saved" => candidate_only,
        "paper_ready_solver_status" => feasible_certificate_saved,
        "certificate_warning" => feasible_certificate_saved ?
            "Final MOSEK status was accepted as certificate-grade." :
            "Diagnostic TSSOS candidate saved from a non-certificate-grade solver status; validate carefully and do not use as a proof.",
        "num_psatz_constraints" => num_psatz_constraints,
        "wf_decrease" => WF_DEC,
        "C_degree" => C_DEGREE,
        "tssos_orders" => Dict(
            "P1" => TSSOS_ORDER_P1,
            "P2" => TSSOS_ORDER_P2,
            "P3" => TSSOS_ORDER_P3,
        ),
        "tssos_sparsity" => Dict(
            "CS" => TSSOS_CS,
            "TS" => TSSOS_TS,
            "SO" => TSSOS_SO,
            "GroebnerBasis" => TSSOS_USE_GROEBNER,
        ),
        "implication_multipliers" => Dict(
            "IMP_MULT_P2" => IMP_MULT_P2,
            "IMP_MULT_P3_A" => IMP_MULT_P3_A,
            "IMP_MULT_P3_B" => IMP_MULT_P3_B,
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
            "gap_min" => 0.0,
        ),
        "initial_set" => Dict(
            "gap_max" => 0.3,
        ),
        "persistence_set" => Dict(
            "gap_max" => 0.4,
            "interpretation" => "Xu is used as the persistence region in this experiment.",
        ),
        "unsafe_set" => Dict(
            "gap_max" => 0.4,
            "note" => "Kept for backward compatibility with existing validation scripts; in the paper, interpret Xu as the persistence region.",
        ),
        "conditions" => Dict(
            "P1" => "For each edge (p,sigma,q), C[p,q](x,f_sigma(x)) >= 0 on X.",
            "P2" => "For each edge (p,sigma,q) and node r, C[q,r](f_sigma(x),y) >= 0 implies C[p,r](x,y) >= 0 on X x X, using a fixed scalar S-procedure multiplier.",
            "P3" => "For all p,q,r, C[p,q](x0,y) >= 0 and C[q,r](y,z) >= 0 imply C[p,r](x0,z) <= C[p,q](x0,y) - wf_decrease on X0 x Xu x Xu, using fixed scalar S-procedure multipliers.",
        ),
    )

    open(OUT_JSON, "w") do io
        JSON.print(io, json_dict, 2)
    end

    if feasible_certificate_saved
        println("\nSaved feasible TSSOS certificate results:")
    else
        println("\nSaved diagnostic TSSOS candidate results:")
    end
    println("  ", OUT_JLD2)
    println("  ", OUT_JSON)
end

function main()
    println()
    println("============================================================")
    println("Starting TSSOS PC-CC synthesis")
    println("============================================================")

    model = make_model_with_mosek()
    coeff, C, constraint_infos, num_psatz_constraints = build_tssos_problem(model)

    println("\nNumber of TSSOS Putinar constraints = ", num_psatz_constraints)
    println("Expected count for full implication model = 20")
    println("Starting MOSEK optimization...")

    optimize!(model)

    status = termination_status(model)
    pstatus = primal_status(model)

    println("\nSolver = mosek")
    println("Termination status = ", status)
    println("Primal status      = ", pstatus)

    if has_values(model)
        println("Objective value    = ", safe_objective_value(model))
    else
        println("Objective value    = unavailable because no primal values were returned.")
    end

    if !acceptable_solution(model)
        println("\n[WARN] TSSOS synthesis did not return an accepted feasible certificate.")
        println("       Termination status = ", termination_status(model))
        println("       Primal status      = ", primal_status(model))

        if SAVE_DIAGNOSTIC_CANDIDATE && has_values(model)
            println("       MOSEK returned primal values; saving a diagnostic candidate for validation only.")
            try
                coeff_values = extract_coefficients(coeff)
                save_results(model, coeff_values, num_psatz_constraints; candidate_only=true)
                println("\nNext diagnostic validation command:")
                println("  julia --project=. sos/pccc_validate_platoon_sos.jl ", OUT_JSON, " 21 41")
            catch err
                println("       Diagnostic coefficient extraction failed.")
                println("Error:")
                println(err)
                save_status_only(model, num_psatz_constraints)
            end
        else
            println("       Skipping coefficient extraction and certificate JSON/JLD2 save.")
            save_status_only(model, num_psatz_constraints)
        end

        println("\nDone. No certificate-grade TSSOS result was saved.")
        return
    end

    println("\nAccepted TSSOS certificate found. Extracting coefficients...")
    coeff_values = extract_coefficients(coeff)
    save_results(model, coeff_values, num_psatz_constraints; candidate_only=false)

    println("\nNext validation command:")
    println("  julia --project=. sos/pccc_validate_platoon_sos.jl ", OUT_JSON, " 21 41")
    println("\nDone.")
end

main()
