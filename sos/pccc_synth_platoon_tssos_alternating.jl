# Author: Reza Iraji
# PC-CC / alternating TSSOS synthesis for the two-car platoon example.
#
# This script ports the alternating fixed-C / fixed-multiplier scheme to TSSOS.
# It is intended to replace the failed fixed-scalar TSSOS prototype.
#
# Convex subproblems:
#   Bootstrap Phase B: fixed scalar premise multipliers, solve C.
#   Phase A: fixed C, solve polynomial nonnegative premise multipliers.
#   Phase B: fixed premise multipliers, solve C.
#
# Each implication is encoded by TSSOS.add_psatz! over the relevant
# semialgebraic domain. Polynomial premise multipliers are constrained
# nonnegative on the same domain by additional Putinar constraints.
#
# Run from repository root:
#   julia --project=. sos/pccc_synth_platoon_tssos_alternating.jl
#
# Validate saved JSON, if present:
#   julia --project=. sos/pccc_validate_platoon_sos.jl results/pccc_synth_platoon_tssos_alternating.json 21 41

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

const WF_DEC = 1.0e-6
const C_DEGREE = 2

# Alternating settings.
const ALT_MAX_ITERS = 4
const ALT_C_CONV_TOL = 1.0e-6

# Polynomial premise multiplier degrees. Keep P2 at 0 first because P2 was
# numerically stable in the SumOfSquares alternating runs; give P3 flexibility.
const ALT_P2_PREMISE_MULT_DEG = 0
const ALT_P3_PREMISE_MULT_DEG = 2

# Baseline scalar multipliers used only for bootstrap Phase B.
const IMP_MULT_P2   = 0.001
const IMP_MULT_P3_A = 0.01
const IMP_MULT_P3_B = 0.01

# TSSOS Putinar orders. These are intentionally moderate. Increase P3 first
# if P1/P2 pass but P3 fails validation.
const TSSOS_ORDER_P1 = 2
const TSSOS_ORDER_P2 = 3
const TSSOS_ORDER_P3 = 3
const TSSOS_MULT_NONNEG_ORDER_P2 = 2
const TSSOS_MULT_NONNEG_ORDER_P3 = 2

# TSSOS settings. Start sparse; switch to false/false only if sparse settings
# give conversion errors or obviously bad candidates.
const TSSOS_CS = "MF"
const TSSOS_TS = "block"
const TSSOS_SO = 1
const TSSOS_GROEBNER = false
const TSSOS_QUIET = false

# Numerical regularization. These constraints are not theorem assumptions;
# they are synthesis normalizations to avoid huge numerically meaningless
# diagnostic candidates such as the failed fixed-scalar TSSOS runs.
const USE_COEFF_BOUNDS = true
const C_COEFF_BOUND = 1.0e4
const MULT_COEFF_BOUND = 1.0e4
const USE_L1_OBJECTIVE = true

# Candidate extraction policy. Approximate iterates may be used to continue
# alternating, but only accepted solver statuses are marked paper-ready.
const SAVE_DIAGNOSTIC_CANDIDATE = true

# Phase A is a multiplier-learning subproblem. P1 has no premise multipliers,
# so enforcing P1 in Phase A can kill the loop when the bootstrap C is only
# approximate. Phase B remains responsible for enforcing P1 on the new C.
const ENFORCE_P1_IN_PHASEA = false

# Phase A may be softened to obtain useful multiplier iterates from a rough
# bootstrap C. These slacks are internal to Phase A and are never saved as
# part of a certificate; Phase B and final validation remain hard.
const USE_PHASEA_SOFT_IMPLICATION_SLACKS = true
const PHASEA_SLACK_PENALTY = 1.0e6

const RESULTS_DIR = "results"
const OUT_JLD2 = joinpath(RESULTS_DIR, "pccc_synth_platoon_tssos_alternating.jld2")
const OUT_JSON = joinpath(RESULTS_DIR, "pccc_synth_platoon_tssos_alternating.json")
const OUT_STATUS_JSON = joinpath(RESULTS_DIR, "pccc_synth_platoon_tssos_alternating_status.json")

const USE_SILENT_SOLVER = false

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

const APPROX_TERMS = [
    MOI.SLOW_PROGRESS,
    MOI.ITERATION_LIMIT,
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
# Semialgebraic domains. A list [g1,...,gm] means gi >= 0.
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

# In P3 we reuse x1,x2 as the local dummy variables for x0.
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
# Template basis for C[p,q](x,y)
# ---------------------------------------------------------------------

basis_xy = monomials([x1, x2, y1, y2], 0:C_DEGREE)
nbasis = length(basis_xy)

println("Using alternating TSSOS synthesis (v11 save-all-iterates, v4 mathematics).")
println("Using quadratic template for C[p,q](x,y).")
println("Number of basis monomials = ", nbasis)
println("P2 premise multiplier degree = ", ALT_P2_PREMISE_MULT_DEG)
println("P3 premise multiplier degree = ", ALT_P3_PREMISE_MULT_DEG)
println("TSSOS orders: P1=", TSSOS_ORDER_P1, ", P2=", TSSOS_ORDER_P2, ", P3=", TSSOS_ORDER_P3)
println("TSSOS settings: CS=", TSSOS_CS, ", TS=", TSSOS_TS, ", SO=", TSSOS_SO, ", GroebnerBasis=", TSSOS_GROEBNER)

# ---------------------------------------------------------------------
# Solver and TSSOS helpers
# ---------------------------------------------------------------------

function make_model_with_mosek()
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    if USE_SILENT_SOLVER
        set_silent(model)
    end
    return model
end

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
        GroebnerBasis = TSSOS_GROEBNER,
    )
    return info
end

function l1_bound_expr!(model, vars; bound=Inf, basename="l1")
    v = collect(vars)
    if isempty(v)
        return 0.0
    end
    t = @variable(model, [1:length(v)], lower_bound=0.0, base_name=string(gensym(Symbol(basename))))
    for i in 1:length(v)
        @constraint(model, t[i] >= v[i])
        @constraint(model, t[i] >= -v[i])
        if isfinite(bound)
            @constraint(model, v[i] <= bound)
            @constraint(model, v[i] >= -bound)
        end
    end
    return sum(t)
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
# Numeric/fixed C helpers
# ---------------------------------------------------------------------

function fixed_C_poly(C_coeff_fixed, p, q)
    return sum(C_coeff_fixed[(p, q, k)] * basis_xy[k] for k in 1:nbasis)
end

function extract_C_coeffs(coeff)
    Cc = Dict{Tuple{Int, Int, Int}, Float64}()
    for p in nodes
        for q in nodes
            for k in 1:nbasis
                val = value(coeff[p, q, k])
                if !(val isa Number) || !isfinite(val)
                    error("Invalid C coefficient value at (p,q,k)=($p,$q,$k): $val")
                end
                Cc[(p, q, k)] = Float64(val)
            end
        end
    end
    return Cc
end

function coeff_dict_to_array(C_coeff)
    coeff_values = zeros(length(nodes), length(nodes), nbasis)
    for p in nodes
        for q in nodes
            for k in 1:nbasis
                coeff_values[p, q, k] = C_coeff[(p, q, k)]
            end
        end
    end
    return coeff_values
end

function max_C_coeff_change(C_old, C_new)
    delta = 0.0
    for key in keys(C_old)
        delta = max(delta, abs(C_new[key] - C_old[key]))
    end
    return delta
end

# ---------------------------------------------------------------------
# Multiplier containers
# ---------------------------------------------------------------------

mutable struct MultiplierDecision
    poly::Any
    coeffs::Any
    basis::Any
end

mutable struct MultiplierFixed
    coeffs::Vector{Float64}
    basis::Any
end

mutable struct MultiplierDecisionStore
    p2::Dict{Tuple{Int, Int, Int, Int}, MultiplierDecision}
    p3a::Dict{Tuple{Int, Int, Int}, MultiplierDecision}
    p3b::Dict{Tuple{Int, Int, Int}, MultiplierDecision}
end

mutable struct MultiplierFixedStore
    p2::Dict{Tuple{Int, Int, Int, Int}, MultiplierFixed}
    p3a::Dict{Tuple{Int, Int, Int}, MultiplierFixed}
    p3b::Dict{Tuple{Int, Int, Int}, MultiplierFixed}
end

function empty_multiplier_decision_store()
    return MultiplierDecisionStore(
        Dict{Tuple{Int, Int, Int, Int}, MultiplierDecision}(),
        Dict{Tuple{Int, Int, Int}, MultiplierDecision}(),
        Dict{Tuple{Int, Int, Int}, MultiplierDecision}(),
    )
end

function empty_multiplier_fixed_store()
    return MultiplierFixedStore(
        Dict{Tuple{Int, Int, Int, Int}, MultiplierFixed}(),
        Dict{Tuple{Int, Int, Int}, MultiplierFixed}(),
        Dict{Tuple{Int, Int, Int}, MultiplierFixed}(),
    )
end

function new_multiplier_decision!(model, vars, degree; basename="mult")
    poly, coeffs, basis = add_poly!(model, vars, degree)
    return MultiplierDecision(poly, coeffs, basis)
end

function fixed_multiplier_poly(mult::MultiplierFixed)
    return sum(mult.coeffs[i] * mult.basis[i] for i in 1:length(mult.basis))
end

function extract_fixed_multiplier(dec::MultiplierDecision)
    vals = Float64[]
    for i in 1:length(dec.coeffs)
        val = value(dec.coeffs[i])
        if !(val isa Number) || !isfinite(val)
            error("Invalid multiplier coefficient value: $val")
        end
        push!(vals, Float64(val))
    end
    return MultiplierFixed(vals, dec.basis)
end

function extract_fixed_multipliers(dec_store::MultiplierDecisionStore)
    fixed = empty_multiplier_fixed_store()
    for (key, dec) in dec_store.p2
        fixed.p2[key] = extract_fixed_multiplier(dec)
    end
    for (key, dec) in dec_store.p3a
        fixed.p3a[key] = extract_fixed_multiplier(dec)
    end
    for (key, dec) in dec_store.p3b
        fixed.p3b[key] = extract_fixed_multiplier(dec)
    end
    return fixed
end

function fixed_p2_multiplier_poly_or_default(mult_fixed, key)
    if mult_fixed === nothing || !haskey(mult_fixed.p2, key)
        return IMP_MULT_P2
    end
    return fixed_multiplier_poly(mult_fixed.p2[key])
end

function fixed_p3a_multiplier_poly_or_default(mult_fixed, key)
    if mult_fixed === nothing || !haskey(mult_fixed.p3a, key)
        return IMP_MULT_P3_A
    end
    return fixed_multiplier_poly(mult_fixed.p3a[key])
end

function fixed_p3b_multiplier_poly_or_default(mult_fixed, key)
    if mult_fixed === nothing || !haskey(mult_fixed.p3b, key)
        return IMP_MULT_P3_B
    end
    return fixed_multiplier_poly(mult_fixed.p3b[key])
end

function serialize_fixed_multiplier_store(mult_fixed)
    if mult_fixed === nothing
        return nothing
    end
    p2 = Dict{String, Any}()
    p3a = Dict{String, Any}()
    p3b = Dict{String, Any}()
    for (key, mult) in mult_fixed.p2
        p2[string(key)] = Dict("coeffs" => mult.coeffs, "basis" => [string(b) for b in mult.basis])
    end
    for (key, mult) in mult_fixed.p3a
        p3a[string(key)] = Dict("coeffs" => mult.coeffs, "basis" => [string(b) for b in mult.basis])
    end
    for (key, mult) in mult_fixed.p3b
        p3b[string(key)] = Dict("coeffs" => mult.coeffs, "basis" => [string(b) for b in mult.basis])
    end
    return Dict("p2" => p2, "p3a" => p3a, "p3b" => p3b)
end

# ---------------------------------------------------------------------
# Phase A: fixed C, solve polynomial premise multipliers
# ---------------------------------------------------------------------

function phaseA_implication_slack!(model, objective_terms; basename="phaseA_slack")
    if !USE_PHASEA_SOFT_IMPLICATION_SLACKS
        return 0.0
    end

    slack = @variable(
        model,
        lower_bound = 0.0,
        base_name = string(gensym(Symbol(basename)))
    )
    push!(objective_terms, PHASEA_SLACK_PENALTY * slack)
    return slack
end

function build_phaseA_problem(model, C_coeff_fixed)
    C_fixed = Matrix{Any}(undef, length(nodes), length(nodes))
    for p in nodes
        for q in nodes
            C_fixed[p, q] = fixed_C_poly(C_coeff_fixed, p, q)
        end
    end

    mult_decisions = empty_multiplier_decision_store()
    objective_terms = Any[]
    num_psatz_constraints = 0

    # P1 has no premise multipliers. In Phase A we normally skip it so that
    # a rough bootstrap C can still be used to learn P2/P3 multipliers. Phase B
    # enforces P1 hard on the new C iterate.
    if ENFORCE_P1_IN_PHASEA
        for (p, sigma, q) in edges
            poly = C_x_fx(C_fixed[p, q], sigma)
            add_putinar_nonneg!(model, poly, [x1, x2], X_cons_x, TSSOS_ORDER_P1;
                label="A_P1_edge_$(p)_$(sigma)_$(q)")
            num_psatz_constraints += 1
        end
    else
        println("Phase A: skipping P1 constraints because P1 has no premise multipliers.")
    end

    XY_cons = vcat(X_cons_x, X_cons_y)
    vars_p2 = [x1, x2, y1, y2]

    for (p, sigma, q) in edges
        for r in nodes
            key = (p, sigma, q, r)
            premise = C_fx_y(C_fixed[q, r], sigma)
            conclusion = C_x_y(C_fixed[p, r])
            s_dec = new_multiplier_decision!(model, vars_p2, ALT_P2_PREMISE_MULT_DEG;
                basename="p2mult")
            mult_decisions.p2[key] = s_dec

            # Enforce s >= 0 on X x X, then implication certificate.
            add_putinar_nonneg!(model, s_dec.poly, vars_p2, XY_cons, TSSOS_MULT_NONNEG_ORDER_P2;
                label="A_P2_mult_nonneg_$(p)_$(sigma)_$(q)_$(r)")
            num_psatz_constraints += 1
            p2_slack = phaseA_implication_slack!(model, objective_terms;
                basename="A_P2_slack")
            add_putinar_nonneg!(model, conclusion - s_dec.poly * premise + p2_slack,
                vars_p2, XY_cons, TSSOS_ORDER_P2;
                label="A_P2_imp_$(p)_$(sigma)_$(q)_$(r)")
            num_psatz_constraints += 1

            push!(objective_terms, l1_bound_expr!(model, s_dec.coeffs;
                bound=(USE_COEFF_BOUNDS ? MULT_COEFF_BOUND : Inf), basename="p2_l1"))
        end
    end

    X0_Xu_Xu_cons = vcat(X0_cons_x, Xu_cons_y, Xu_cons_z)
    vars_p3 = [x1, x2, y1, y2, z1, z2]

    for p in nodes
        for q in nodes
            for r in nodes
                key = (p, q, r)
                premise1 = C_x0_y(C_fixed[p, q])
                premise2 = C_y_z(C_fixed[q, r])
                conclusion = C_x0_y(C_fixed[p, q]) - WF_DEC - C_x0_z(C_fixed[p, r])

                s1_dec = new_multiplier_decision!(model, vars_p3, ALT_P3_PREMISE_MULT_DEG;
                    basename="p3amult")
                s2_dec = new_multiplier_decision!(model, vars_p3, ALT_P3_PREMISE_MULT_DEG;
                    basename="p3bmult")
                mult_decisions.p3a[key] = s1_dec
                mult_decisions.p3b[key] = s2_dec

                add_putinar_nonneg!(model, s1_dec.poly, vars_p3, X0_Xu_Xu_cons, TSSOS_MULT_NONNEG_ORDER_P3;
                    label="A_P3A_mult_nonneg_$(p)_$(q)_$(r)")
                num_psatz_constraints += 1
                add_putinar_nonneg!(model, s2_dec.poly, vars_p3, X0_Xu_Xu_cons, TSSOS_MULT_NONNEG_ORDER_P3;
                    label="A_P3B_mult_nonneg_$(p)_$(q)_$(r)")
                num_psatz_constraints += 1

                p3_slack = phaseA_implication_slack!(model, objective_terms;
                    basename="A_P3_slack")
                add_putinar_nonneg!(model,
                    conclusion - s1_dec.poly * premise1 - s2_dec.poly * premise2 + p3_slack,
                    vars_p3,
                    X0_Xu_Xu_cons,
                    TSSOS_ORDER_P3;
                    label="A_P3_imp_$(p)_$(q)_$(r)")
                num_psatz_constraints += 1

                push!(objective_terms, l1_bound_expr!(model, s1_dec.coeffs;
                    bound=(USE_COEFF_BOUNDS ? MULT_COEFF_BOUND : Inf), basename="p3a_l1"))
                push!(objective_terms, l1_bound_expr!(model, s2_dec.coeffs;
                    bound=(USE_COEFF_BOUNDS ? MULT_COEFF_BOUND : Inf), basename="p3b_l1"))
            end
        end
    end

    if USE_L1_OBJECTIVE
        @objective(model, Min, sum(objective_terms))
    else
        @objective(model, Min, 0.0)
    end

    return mult_decisions, num_psatz_constraints
end

# ---------------------------------------------------------------------
# Phase B: fixed multipliers, solve C
# ---------------------------------------------------------------------

function build_phaseB_problem(model, mult_fixed)
    @variable(model, coeff[p in nodes, q in nodes, k in 1:nbasis])

    C = Matrix{Any}(undef, length(nodes), length(nodes))
    for p in nodes
        for q in nodes
            C[p, q] = sum(coeff[p, q, k] * basis_xy[k] for k in 1:nbasis)
        end
    end

    objective_terms = Any[]
    push!(objective_terms, l1_bound_expr!(model, [coeff[p, q, k] for p in nodes for q in nodes for k in 1:nbasis];
        bound=(USE_COEFF_BOUNDS ? C_COEFF_BOUND : Inf), basename="C_l1"))

    num_psatz_constraints = 0

    for (p, sigma, q) in edges
        poly = C_x_fx(C[p, q], sigma)
        add_putinar_nonneg!(model, poly, [x1, x2], X_cons_x, TSSOS_ORDER_P1;
            label="B_P1_edge_$(p)_$(sigma)_$(q)")
        num_psatz_constraints += 1
    end

    XY_cons = vcat(X_cons_x, X_cons_y)
    for (p, sigma, q) in edges
        for r in nodes
            key = (p, sigma, q, r)
            premise = C_fx_y(C[q, r], sigma)
            conclusion = C_x_y(C[p, r])
            s_fixed = fixed_p2_multiplier_poly_or_default(mult_fixed, key)
            add_putinar_nonneg!(model, conclusion - s_fixed * premise,
                [x1, x2, y1, y2], XY_cons, TSSOS_ORDER_P2;
                label="B_P2_imp_$(p)_$(sigma)_$(q)_$(r)")
            num_psatz_constraints += 1
        end
    end

    X0_Xu_Xu_cons = vcat(X0_cons_x, Xu_cons_y, Xu_cons_z)
    for p in nodes
        for q in nodes
            for r in nodes
                key = (p, q, r)
                premise1 = C_x0_y(C[p, q])
                premise2 = C_y_z(C[q, r])
                conclusion = C_x0_y(C[p, q]) - WF_DEC - C_x0_z(C[p, r])
                s1_fixed = fixed_p3a_multiplier_poly_or_default(mult_fixed, key)
                s2_fixed = fixed_p3b_multiplier_poly_or_default(mult_fixed, key)
                add_putinar_nonneg!(model,
                    conclusion - s1_fixed * premise1 - s2_fixed * premise2,
                    [x1, x2, y1, y2, z1, z2], X0_Xu_Xu_cons, TSSOS_ORDER_P3;
                    label="B_P3_imp_$(p)_$(q)_$(r)")
                num_psatz_constraints += 1
            end
        end
    end

    if USE_L1_OBJECTIVE
        @objective(model, Min, sum(objective_terms))
    else
        @objective(model, Min, 0.0)
    end

    return coeff, C, num_psatz_constraints
end

# ---------------------------------------------------------------------
# Solve utilities
# ---------------------------------------------------------------------

function acceptable_solution(model)
    return termination_status(model) in ACCEPT_TERMS &&
           primal_status(model) in ACCEPT_PRIMALS &&
           has_values(model)
end

function acceptable_iterate(model)
    if acceptable_solution(model)
        return true
    end
    return has_values(model) && termination_status(model) in APPROX_TERMS
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
    end
    return nothing
end

function print_solve_summary(model, phase_name)
    println("\nPhase = ", phase_name)
    println("Termination status = ", termination_status(model))
    println("Primal status      = ", primal_status(model))
    if has_values(model)
        println("Objective value    = ", safe_objective_value(model))
    else
        println("Objective value    = unavailable because no primal values were returned.")
    end
end

function solve_phaseA(C_coeff_fixed)
    println("\nBuilding Phase A fixed-C multiplier model with TSSOS/MOSEK...")
    model = make_model_with_mosek()
    mult_decisions, num_psatz_constraints = build_phaseA_problem(model, C_coeff_fixed)
    println("Number of TSSOS Putinar constraints = ", num_psatz_constraints)
    println("Starting Phase A optimization...")
    optimize!(model)
    print_solve_summary(model, "Phase A")

    if acceptable_iterate(model)
        if !acceptable_solution(model)
            println("[WARN] Phase A returned only an approximate iterate. It may be used for alternating, not as proof.")
        end
        mult_fixed = extract_fixed_multipliers(mult_decisions)
        return mult_fixed, model, num_psatz_constraints
    end
    return nothing, model, num_psatz_constraints
end

function solve_phaseB(mult_fixed; seed_phase=false)
    println("\nBuilding Phase B fixed-multiplier C model with TSSOS/MOSEK...")
    model = make_model_with_mosek()
    coeff, C, num_psatz_constraints = build_phaseB_problem(model, mult_fixed)
    println("Number of TSSOS Putinar constraints = ", num_psatz_constraints)
    if mult_fixed === nothing
        println("Fixed multipliers = baseline scalar bootstrap multipliers")
    else
        println("Fixed P2 multipliers = ", length(mult_fixed.p2))
        println("Fixed P3A multipliers = ", length(mult_fixed.p3a))
        println("Fixed P3B multipliers = ", length(mult_fixed.p3b))
    end
    println("Starting Phase B optimization...")
    optimize!(model)
    print_solve_summary(model, seed_phase ? "Bootstrap Phase B" : "Phase B")

    if acceptable_iterate(model)
        if !acceptable_solution(model)
            println("[WARN] Phase B returned only an approximate C iterate. It may be used for alternating, not as proof.")
        end
        C_coeff = extract_C_coeffs(coeff)
        return C_coeff, model, num_psatz_constraints
    end
    return nothing, model, num_psatz_constraints
end

# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

function save_status_only(status_dict)
    mkpath(RESULTS_DIR)
    open(OUT_STATUS_JSON, "w") do io
        JSON.print(io, status_dict, 2)
    end
    println("\nSaved alternating TSSOS status summary:")
    println("  ", OUT_STATUS_JSON)
end


function save_alternating_json_to(out_json::String, C_coeff, best_mult, last_phaseB_model, iterations_completed, total_phaseA_psatz, total_phaseB_psatz)
    mkpath(dirname(out_json))

    coeff_values = coeff_dict_to_array(C_coeff)
    basis_strings = [string(b) for b in basis_xy]
    edge_array = [[p, sigma, q] for (p, sigma, q) in edges]
    status_string = last_phaseB_model === nothing ? "UNKNOWN" : string(termination_status(last_phaseB_model))
    primal_status_string = last_phaseB_model === nothing ? "UNKNOWN" : string(primal_status(last_phaseB_model))
    objective = last_phaseB_model === nothing ? nothing : safe_objective_value(last_phaseB_model)
    final_solver_accepted = last_phaseB_model !== nothing && acceptable_solution(last_phaseB_model)
    certificate_type = "tssos_alternating_implication_pairwise_polynomial_multipliers"
    serialized_multipliers = serialize_fixed_multiplier_store(best_mult)

    json_dict = Dict(
        "description" => "Per-iteration alternating TSSOS implication-style PC-CC synthesis result for two-car platoon example",
        "certificate_type" => certificate_type,
        "status" => status_string,
        "primal_status" => primal_status_string,
        "objective" => objective,
        "iterations_completed" => iterations_completed,
        "feasible_certificate_saved" => final_solver_accepted,
        "candidate_saved" => !final_solver_accepted,
        "paper_ready_solver_status" => final_solver_accepted,
        "certificate_warning" => final_solver_accepted ?
            "Final Phase B solver status was accepted as certificate-grade." :
            "Per-iteration candidate saved from an approximate TSSOS alternating iterate; validate carefully and do not use as a proof without certificate-grade solver status.",
        "wf_decrease" => WF_DEC,
        "C_degree" => C_DEGREE,
        "P2_premise_multiplier_degree" => ALT_P2_PREMISE_MULT_DEG,
        "P3_premise_multiplier_degree" => ALT_P3_PREMISE_MULT_DEG,
        "tssos_orders" => Dict("P1" => TSSOS_ORDER_P1, "P2" => TSSOS_ORDER_P2, "P3" => TSSOS_ORDER_P3),
        "tssos_multiplier_nonneg_orders" => Dict("P2" => TSSOS_MULT_NONNEG_ORDER_P2, "P3" => TSSOS_MULT_NONNEG_ORDER_P3),
        "tssos_sparsity" => Dict("CS" => TSSOS_CS, "TS" => TSSOS_TS, "SO" => TSSOS_SO, "GroebnerBasis" => TSSOS_GROEBNER),
        "coefficient_regularization" => Dict("USE_COEFF_BOUNDS" => USE_COEFF_BOUNDS, "C_COEFF_BOUND" => C_COEFF_BOUND, "MULT_COEFF_BOUND" => MULT_COEFF_BOUND, "USE_L1_OBJECTIVE" => USE_L1_OBJECTIVE),
        "total_phaseA_psatz_constraints" => total_phaseA_psatz,
        "total_phaseB_psatz_constraints" => total_phaseB_psatz,
        "fixed_premise_multipliers" => serialized_multipliers,
        "bootstrap_fixed_scalar_multipliers" => Dict("IMP_MULT_P2" => IMP_MULT_P2, "IMP_MULT_P3_A" => IMP_MULT_P3_A, "IMP_MULT_P3_B" => IMP_MULT_P3_B),
        "nodes" => nodes,
        "edges" => edge_array,
        "basis_xy" => basis_strings,
        "coeff_values" => coeff_values,
        "state_domain" => Dict("x1_min" => 0.0, "x1_max" => 3.0, "x2_min" => 0.0, "x2_max" => 5.1, "gap_min" => 0.0),
        "initial_set" => Dict("gap_max" => 0.3),
        "persistence_set" => Dict("gap_max" => 0.4, "interpretation" => "Xu is used as the persistence region in this experiment."),
        "unsafe_set" => Dict("gap_max" => 0.4, "note" => "Kept for backward compatibility with existing validation scripts; in the paper, interpret Xu as the persistence region."),
        "conditions" => Dict(
            "P1" => "For each edge (p,sigma,q), C[p,q](x,f_sigma(x)) >= 0 on X.",
            "P2" => "For each edge (p,sigma,q) and node r, C[q,r](f_sigma(x),y) >= 0 implies C[p,r](x,y) >= 0 on X x X, using nonnegative polynomial premise multipliers.",
            "P3" => "For all p,q,r, C[p,q](x0,y) >= 0 and C[q,r](y,z) >= 0 imply C[p,r](x0,z) <= C[p,q](x0,y) - wf_decrease on X0 x Xu x Xu, using nonnegative polynomial premise multipliers."
        ),
    )

    open(out_json, "w") do io
        JSON.print(io, json_dict, 2)
    end
    println("Saved per-iteration diagnostic JSON: ", out_json)
end

function save_alternating_results(C_coeff, best_mult, last_phaseB_model, iterations_completed, total_phaseA_psatz, total_phaseB_psatz)
    mkpath(RESULTS_DIR)

    coeff_values = coeff_dict_to_array(C_coeff)
    basis_strings = [string(b) for b in basis_xy]
    edge_array = [[p, sigma, q] for (p, sigma, q) in edges]
    status_string = last_phaseB_model === nothing ? "UNKNOWN" : string(termination_status(last_phaseB_model))
    primal_status_string = last_phaseB_model === nothing ? "UNKNOWN" : string(primal_status(last_phaseB_model))
    objective = last_phaseB_model === nothing ? nothing : safe_objective_value(last_phaseB_model)
    final_solver_accepted = last_phaseB_model !== nothing && acceptable_solution(last_phaseB_model)
    certificate_type = "tssos_alternating_implication_pairwise_polynomial_multipliers"
    serialized_multipliers = serialize_fixed_multiplier_store(best_mult)

    @save OUT_JLD2 coeff_values basis_strings edge_array WF_DEC status_string primal_status_string objective certificate_type iterations_completed serialized_multipliers final_solver_accepted

    json_dict = Dict(
        "description" => "Alternating TSSOS implication-style PC-CC synthesis result for two-car platoon example",
        "certificate_type" => certificate_type,
        "status" => status_string,
        "primal_status" => primal_status_string,
        "objective" => objective,
        "iterations_completed" => iterations_completed,
        "feasible_certificate_saved" => final_solver_accepted,
        "candidate_saved" => !final_solver_accepted,
        "paper_ready_solver_status" => final_solver_accepted,
        "certificate_warning" => final_solver_accepted ?
            "Final Phase B solver status was accepted as certificate-grade." :
            "Candidate saved from an approximate TSSOS alternating iterate; validate carefully and do not use as a proof without certificate-grade solver status.",
        "wf_decrease" => WF_DEC,
        "C_degree" => C_DEGREE,
        "P2_premise_multiplier_degree" => ALT_P2_PREMISE_MULT_DEG,
        "P3_premise_multiplier_degree" => ALT_P3_PREMISE_MULT_DEG,
        "tssos_orders" => Dict("P1" => TSSOS_ORDER_P1, "P2" => TSSOS_ORDER_P2, "P3" => TSSOS_ORDER_P3),
        "tssos_multiplier_nonneg_orders" => Dict("P2" => TSSOS_MULT_NONNEG_ORDER_P2, "P3" => TSSOS_MULT_NONNEG_ORDER_P3),
        "tssos_sparsity" => Dict("CS" => TSSOS_CS, "TS" => TSSOS_TS, "SO" => TSSOS_SO, "GroebnerBasis" => TSSOS_GROEBNER),
        "coefficient_regularization" => Dict("USE_COEFF_BOUNDS" => USE_COEFF_BOUNDS, "C_COEFF_BOUND" => C_COEFF_BOUND, "MULT_COEFF_BOUND" => MULT_COEFF_BOUND, "USE_L1_OBJECTIVE" => USE_L1_OBJECTIVE),
        "total_phaseA_psatz_constraints" => total_phaseA_psatz,
        "total_phaseB_psatz_constraints" => total_phaseB_psatz,
        "fixed_premise_multipliers" => serialized_multipliers,
        "bootstrap_fixed_scalar_multipliers" => Dict("IMP_MULT_P2" => IMP_MULT_P2, "IMP_MULT_P3_A" => IMP_MULT_P3_A, "IMP_MULT_P3_B" => IMP_MULT_P3_B),
        "nodes" => nodes,
        "edges" => edge_array,
        "basis_xy" => basis_strings,
        "coeff_values" => coeff_values,
        "state_domain" => Dict("x1_min" => 0.0, "x1_max" => 3.0, "x2_min" => 0.0, "x2_max" => 5.1, "gap_min" => 0.0),
        "initial_set" => Dict("gap_max" => 0.3),
        "persistence_set" => Dict("gap_max" => 0.4, "interpretation" => "Xu is used as the persistence region in this experiment."),
        "unsafe_set" => Dict("gap_max" => 0.4, "note" => "Kept for backward compatibility with existing validation scripts; in the paper, interpret Xu as the persistence region."),
        "conditions" => Dict(
            "P1" => "For each edge (p,sigma,q), C[p,q](x,f_sigma(x)) >= 0 on X.",
            "P2" => "For each edge (p,sigma,q) and node r, C[q,r](f_sigma(x),y) >= 0 implies C[p,r](x,y) >= 0 on X x X, using nonnegative polynomial premise multipliers.",
            "P3" => "For all p,q,r, C[p,q](x0,y) >= 0 and C[q,r](y,z) >= 0 imply C[p,r](x0,z) <= C[p,q](x0,y) - wf_decrease on X0 x Xu x Xu, using nonnegative polynomial premise multipliers."
        ),
    )

    open(OUT_JSON, "w") do io
        JSON.print(io, json_dict, 2)
    end

    if final_solver_accepted
        println("\nSaved certificate-grade alternating TSSOS results:")
    else
        println("\nSaved diagnostic alternating TSSOS candidate results:")
    end
    println("  ", OUT_JLD2)
    println("  ", OUT_JSON)
end

# ---------------------------------------------------------------------
# Main alternating loop
# ---------------------------------------------------------------------

function main()
    println()
    println("============================================================")
    println("Starting alternating TSSOS synthesis")
    println("============================================================")

    println("\n=== Bootstrap Phase B: fixed scalar multipliers, solve C ===")
    C_coeff_current, seed_model, seed_psatz = solve_phaseB(nothing; seed_phase=true)

    if C_coeff_current === nothing
        println("\n[ERROR] Bootstrap Phase B failed; no initial C seed was obtained.")
        save_status_only(Dict(
            "description" => "Alternating TSSOS PC-CC synthesis status for two-car platoon example",
            "certificate_type" => "tssos_alternating_implication_pairwise_polynomial_multipliers",
            "feasible_certificate_saved" => false,
            "message" => "Bootstrap Phase B failed; no initial C seed was obtained.",
            "bootstrap_status" => string(termination_status(seed_model)),
            "bootstrap_primal_status" => string(primal_status(seed_model)),
            "bootstrap_psatz_constraints" => seed_psatz,
            "wf_decrease" => WF_DEC,
            "C_degree" => C_DEGREE,
            "P2_premise_multiplier_degree" => ALT_P2_PREMISE_MULT_DEG,
            "P3_premise_multiplier_degree" => ALT_P3_PREMISE_MULT_DEG,
        ))
        println("\nDone, but no alternating TSSOS candidate was saved.")
        return
    end

    println("\n[OK] Bootstrap Phase B produced an initial C seed.")
    println("     bootstrap termination status = ", termination_status(seed_model))
    println("     bootstrap primal status      = ", primal_status(seed_model))

    best_C_coeff = nothing
    best_mult = nothing
    last_phaseB_model = nothing
    iterations_completed = 0
    total_phaseA_psatz = 0
    total_phaseB_psatz = seed_psatz

    for iter in 1:ALT_MAX_ITERS
        println()
        println("============================================================")
        println("Alternating iteration ", iter, " / ", ALT_MAX_ITERS)
        println("============================================================")

        println("\n=== Phase A: fixed C, solve polynomial premise multipliers ===")
        mult_fixed, phaseA_model, phaseA_psatz = solve_phaseA(C_coeff_current)
        total_phaseA_psatz += phaseA_psatz

        if mult_fixed === nothing
            println("\n[WARN] Phase A failed at iteration ", iter, ".")
            break
        end

        println("\n[OK] Phase A produced multiplier set.")
        println("Extracted P2 multipliers  = ", length(mult_fixed.p2))
        println("Extracted P3A multipliers = ", length(mult_fixed.p3a))
        println("Extracted P3B multipliers = ", length(mult_fixed.p3b))

        println("\n=== Phase B: fixed multipliers, solve C ===")
        C_coeff_new, phaseB_model, phaseB_psatz = solve_phaseB(mult_fixed; seed_phase=false)
        total_phaseB_psatz += phaseB_psatz

        if C_coeff_new === nothing
            println("\n[WARN] Phase B failed at iteration ", iter, ".")
            break
        end

        delta_C = max_C_coeff_change(C_coeff_current, C_coeff_new)
        println("\n[OK] Phase B produced new C iterate.")
        println("Max C coefficient change = ", delta_C)

        best_C_coeff = C_coeff_new
        best_mult = mult_fixed
        last_phaseB_model = phaseB_model
        iterations_completed = iter

        iter_json = joinpath(RESULTS_DIR, "pccc_synth_platoon_tssos_alternating_iter$(iter).json")
        save_alternating_json_to(iter_json, C_coeff_new, mult_fixed, phaseB_model, iter, total_phaseA_psatz, total_phaseB_psatz)

        C_coeff_current = C_coeff_new

        if delta_C < ALT_C_CONV_TOL
            println("\n[OK] Alternating scheme converged by coefficient change.")
            break
        end
    end

    println()
    println("============================================================")
    println("Alternating TSSOS loop finished")
    println("============================================================")
    println("Iterations completed = ", iterations_completed)

    if best_C_coeff === nothing
        println("\n[WARN] No complete Phase A/Phase B iteration succeeded.")
        save_status_only(Dict(
            "description" => "Alternating TSSOS PC-CC synthesis status for two-car platoon example",
            "certificate_type" => "tssos_alternating_implication_pairwise_polynomial_multipliers",
            "feasible_certificate_saved" => false,
            "message" => "No result was saved because no complete Phase A/Phase B iteration succeeded.",
            "iterations_completed" => iterations_completed,
            "wf_decrease" => WF_DEC,
            "C_degree" => C_DEGREE,
            "P2_premise_multiplier_degree" => ALT_P2_PREMISE_MULT_DEG,
            "P3_premise_multiplier_degree" => ALT_P3_PREMISE_MULT_DEG,
        ))
        println("\nDone, but no alternating TSSOS candidate was saved.")
        return
    end

    save_alternating_results(best_C_coeff, best_mult, last_phaseB_model, iterations_completed, total_phaseA_psatz, total_phaseB_psatz)

    println("\nNext validation command:")
    println("  julia --project=. sos/pccc_validate_platoon_sos.jl ", OUT_JSON, " 21 41")
    println("\nDone.")
end

main()
