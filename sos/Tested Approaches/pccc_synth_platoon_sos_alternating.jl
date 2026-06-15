# Author: Reza Iraji
# PC-CC / alternating SOS synthesis for the two-car platoon example.
#
# This script keeps the pairwise closure certificate C[p,q](x,y) quadratic
# and uses an alternating convex scheme for SOS premise multipliers:
#
#   Phase A: fix C, solve SOS premise multipliers.
#   Phase B: fix SOS premise multipliers, solve C.
#
# This avoids the bilinear one-shot product:
#
#   s_premise * premise(C)
#
# where both s_premise and C would otherwise be decision variables.
#
# Run from the repository root with:
#
#   julia --project=. sos/pccc_synth_platoon_sos_alternating.jl

using JuMP
using SumOfSquares
using DynamicPolynomials
using MultivariatePolynomials
using MosekTools
using SCS
using JLD2
using JSON

const MOI = JuMP.MOI

# ---------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------

# Strict ranking decrease in the well-foundedness implication.
const WF_DEC = 1.0e-6

# Alternating settings.
const ALT_MAX_ITERS = 5
const ALT_P2_PREMISE_MULT_DEG = 0
const ALT_P3_PREMISE_MULT_DEG = 2
const ALT_C_CONV_TOL = 1.0e-6

# --------------------------------------------------------------------------
# Nontriviality / relation-shaping anchors
#
# The relation represented by C[p,q](x,y) >= 0 should not become universal.
# These finite anchor constraints force C[p,q] to be negative at selected
# obviously unrelated pairs. They are synthesis normalizations, not additional
# theorem assumptions.
# --------------------------------------------------------------------------

const ALT_USE_NEGATIVE_ANCHORS = false
const ALT_NEG_ANCHOR_MARGIN = 1.0e-5

const ALT_NEG_ANCHORS = [
    # (a1, a2, b1, b2) means enforce C[p,q]((a1,a2),(b1,b2)) <= -margin.

    # Points inspired by the P3 validation failure: far-apart x/y pairs.
    (3.0, 3.1875, 0.0, 0.0),
    (0.0, 0.0, 3.0, 3.315),
]

# Keep this false in the alternating script. The direct joint SOS-multiplier
# model is bilinear and was already rejected experimentally.
const USE_SOS_PREMISE_MULTIPLIERS = false

# Baseline fixed scalar multipliers. Only kept for metadata and comparison;
# the alternating constraints below do not use these constants.
const IMP_MULT_P2   = 0.001
const IMP_MULT_P3_A = 0.01
const IMP_MULT_P3_B = 0.01

const RESULTS_DIR = "results"
const OUT_JLD2 = joinpath(RESULTS_DIR, "pccc_synth_platoon_sos_alternating.jld2")
const OUT_JSON = joinpath(RESULTS_DIR, "pccc_synth_platoon_sos_alternating.json")
const OUT_STATUS_JSON = joinpath(RESULTS_DIR, "pccc_synth_platoon_sos_alternating_status.json")

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

const ALT_ACCEPT_TERMS = ACCEPT_TERMS

# --------------------------------------------------------------------------
# Seed acceptance policy
#
# For a final certificate, we only accept genuinely successful solver statuses.
# For the bootstrap C seed, however, we allow SCS to provide a rough feasible
# point when it reaches ITERATION_LIMIT with FEASIBLE_POINT. This seed is not
# treated as a certificate; it is only used to initialize the alternating loop.
# --------------------------------------------------------------------------

const ALT_SEED_ACCEPT_TERMS = [
    MOI.OPTIMAL,
    MOI.ALMOST_OPTIMAL,
    MOI.LOCALLY_SOLVED,
    MOI.ALMOST_LOCALLY_SOLVED,
    MOI.ITERATION_LIMIT,
]

function is_acceptable_solution(model)
    return termination_status(model) in ALT_ACCEPT_TERMS
end

function is_acceptable_seed_solution(model)
    term = termination_status(model)
    prim = primal_status(model)

    if !has_values(model)
        return false
    end

    if !(prim in ACCEPT_PRIMALS)
        return false
    end

    # Strictly acceptable seed.
    if term in ALT_ACCEPT_TERMS
        return true
    end

    # Bootstrap only:
    # Allow numerical iterates that have usable primal values.
    # These are not accepted as final certificates.
    if term == MOI.SLOW_PROGRESS
        return true
    end

    if term == MOI.ITERATION_LIMIT
        return true
    end

    return false
end

function is_acceptable_alternating_iterate(model)
    term = termination_status(model)
    prim = primal_status(model)

    if acceptable_solution(model)
        return true
    end

    if !has_values(model)
        return false
    end

    if !(prim in ACCEPT_PRIMALS)
        return false
    end

    # Alternating only:
    # Allow numerical iterates so Phase A can pass multipliers to Phase B.
    # These are not accepted as final paper certificates.
    if term == MOI.SLOW_PROGRESS
        return true
    end

    if term == MOI.ITERATION_LIMIT
        return true
    end

    return false
end

function is_acceptable_phaseB_iterate(model)
    term = termination_status(model)
    prim = primal_status(model)

    if acceptable_solution(model)
        return true
    end

    if !has_values(model)
        return false
    end

    if !(prim in ACCEPT_PRIMALS)
        return false
    end

    # Alternating only:
    # Allow numerical C iterates so the alternating loop can continue.
    # These are not accepted as final paper certificates.
    if term == MOI.SLOW_PROGRESS
        return true
    end

    if term == MOI.ITERATION_LIMIT
        return true
    end

    return false
end

# ---------------------------------------------------------------------
# Polynomial variables
# ---------------------------------------------------------------------

@polyvar x1 x2 y1 y2 z1 z2 x01 x02

const nodes = [1, 2]

# Edge convention: (p, sigma, q) means mode sigma maps source node p to target node q.
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

# ---------------------------------------------------------------------
# Quadratic template basis for C[p,q](x,y)
# ---------------------------------------------------------------------

basis_xy = monomials([x1, x2, y1, y2], 0:2)
nbasis = length(basis_xy)

println("Using alternating SOS synthesis.")
println("Using quadratic template for C[p,q](x,y).")
println("Number of basis monomials = ", nbasis)
println("P2 premise multiplier degree = ", ALT_P2_PREMISE_MULT_DEG)
println("P3 premise multiplier degree = ", ALT_P3_PREMISE_MULT_DEG)
println("Maximum alternating iterations = ", ALT_MAX_ITERS)

# ---------------------------------------------------------------------
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
        "eps_abs" => 1.0e-7,
        "eps_rel" => 1.0e-7,
        "eps_infeas" => 1.0e-8,
        "max_iters" => 500000,
        "verbose" => 1,
    )
    model = SOSModel(optimizer)
    if USE_SILENT_SOLVER
        set_silent(model)
    end
    return model
end

function make_model(solver_name::Symbol)
    if solver_name == :mosek
        return make_model_with_mosek()
    elseif solver_name == :scs
        return make_model_with_scs()
    else
        error("Unknown solver_name = $solver_name")
    end
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
C_x0_y(Cpq) = C_eval(Cpq, x01, x02, y1, y2)
C_x0_z(Cpq) = C_eval(Cpq, x01, x02, z1, z2)
C_y_z(Cpq) = C_eval(Cpq, y1, y2, z1, z2)

# ---------------------------------------------------------------------
# Numeric C helpers
# ---------------------------------------------------------------------

function initial_distance_C_coeffs()
    Cc = Dict{Tuple{Int, Int, Int}, Float64}()

    for p in nodes
        for q in nodes
            for k in 1:nbasis
                Cc[(p, q, k)] = 0.0
            end
        end
    end

    target_terms = Dict(
        x1^2 => 1.0,
        x2^2 => 1.0,
        x1*y1 => -2.0,
        x2*y2 => -2.0,
        y1^2 => 1.0,
        y2^2 => 1.0,
    )

    for p in nodes
        for q in nodes
            for k in 1:nbasis
                mon = basis_xy[k]
                if haskey(target_terms, mon)
                    Cc[(p, q, k)] = target_terms[mon]
                end
            end
        end
    end

    return Cc
end

function fixed_C_poly(C_coeff_fixed, p, q)
    return sum(C_coeff_fixed[(p, q, k)] * basis_xy[k] for k in 1:nbasis)
end

function add_soft_negative_anchor_constraints!(model, C_dec)
    if !ALT_USE_NEGATIVE_ANCHORS
        return 0, 0.0
    end

    nanchors = 0
    anchor_slacks = Any[]

    for p in nodes
        for q in nodes
            for (a1, a2, b1, b2) in ALT_NEG_ANCHORS
                s = @variable(
                    model,
                    lower_bound = 0.0,
                    base_name = string(gensym(:anchor_slack))
                )

                @constraint(
                    model,
                    C_eval(C_dec[p, q], a1, a2, b1, b2) <=
                        -ALT_NEG_ANCHOR_MARGIN + s
                )

                push!(anchor_slacks, s)
                nanchors += 1
            end
        end
    end

    slack_sum =
        isempty(anchor_slacks) ? 0.0 : sum(anchor_slacks)

    return nanchors, slack_sum
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
# SOS multiplier containers and helpers
# ---------------------------------------------------------------------

mutable struct SOSMultiplierDecision
    poly::Any
    coeffs::Any
    basis::Any
end

mutable struct SOSMultiplierFixed
    coeffs::Vector{Float64}
    basis::Any
end

mutable struct MultiplierDecisionStore
    p2::Dict{Tuple{Int, Int, Int, Int}, SOSMultiplierDecision}
    p3a::Dict{Tuple{Int, Int, Int}, SOSMultiplierDecision}
    p3b::Dict{Tuple{Int, Int, Int}, SOSMultiplierDecision}
end

mutable struct MultiplierFixedStore
    p2::Dict{Tuple{Int, Int, Int, Int}, SOSMultiplierFixed}
    p3a::Dict{Tuple{Int, Int, Int}, SOSMultiplierFixed}
    p3b::Dict{Tuple{Int, Int, Int}, SOSMultiplierFixed}
end

function empty_multiplier_decision_store()
    return MultiplierDecisionStore(
        Dict{Tuple{Int, Int, Int, Int}, SOSMultiplierDecision}(),
        Dict{Tuple{Int, Int, Int}, SOSMultiplierDecision}(),
        Dict{Tuple{Int, Int, Int}, SOSMultiplierDecision}(),
    )
end

function empty_multiplier_fixed_store()
    return MultiplierFixedStore(
        Dict{Tuple{Int, Int, Int, Int}, SOSMultiplierFixed}(),
        Dict{Tuple{Int, Int, Int}, SOSMultiplierFixed}(),
        Dict{Tuple{Int, Int, Int}, SOSMultiplierFixed}(),
    )
end

function new_sos_multiplier_decision!(model, vars, degree; basename="sosmult")
    @assert degree >= 0
    @assert iseven(degree) "SOS multiplier degree should be even, e.g. 0 or 2."

    mult_basis = monomials(vars, 0:degree)

    coeffs = @variable(
        model,
        [1:length(mult_basis)],
        base_name = string(gensym(Symbol(basename)))
    )

    s = sum(coeffs[i] * mult_basis[i] for i in 1:length(mult_basis))
    @constraint(model, s in SOSCone())

    return SOSMultiplierDecision(s, coeffs, mult_basis)
end

function fixed_multiplier_poly(mult::SOSMultiplierFixed)
    return sum(mult.coeffs[i] * mult.basis[i] for i in 1:length(mult.basis))
end

function fixed_p2_multiplier_poly_or_default(mult_fixed, key)
    if mult_fixed === nothing
        return IMP_MULT_P2
    end

    if !haskey(mult_fixed.p2, key)
        return IMP_MULT_P2
    end

    return fixed_multiplier_poly(mult_fixed.p2[key])
end

function fixed_p3a_multiplier_poly_or_default(mult_fixed, key)
    if mult_fixed === nothing
        return IMP_MULT_P3_A
    end

    if !haskey(mult_fixed.p3a, key)
        return IMP_MULT_P3_A
    end

    return fixed_multiplier_poly(mult_fixed.p3a[key])
end

function fixed_p3b_multiplier_poly_or_default(mult_fixed, key)
    if mult_fixed === nothing
        return IMP_MULT_P3_B
    end

    if !haskey(mult_fixed.p3b, key)
        return IMP_MULT_P3_B
    end

    return fixed_multiplier_poly(mult_fixed.p3b[key])
end

function extract_fixed_multiplier(dec::SOSMultiplierDecision)
    coeff_vals = Float64[]

    for i in 1:length(dec.coeffs)
        val = value(dec.coeffs[i])

        if !(val isa Number) || !isfinite(val)
            error("Invalid multiplier coefficient value: $val")
        end

        v = Float64(val)

        # For degree-0 SOS multipliers, the single coefficient must be
        # nonnegative. SCS can return tiny negative noise.
        if length(dec.coeffs) == 1 && v < 0.0
            if v >= -1.0e-8
                v = 0.0
            else
                error("Degree-0 SOS multiplier is significantly negative: $v")
            end
        end

        push!(coeff_vals, v)
    end

    return SOSMultiplierFixed(coeff_vals, dec.basis)
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

# ---------------------------------------------------------------------
# SOS/S-procedure domain helpers
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

function add_implication_sos_constraint_phaseA!(
    model,
    conclusion_poly_fixed,
    premise_poly_fixed,
    domain_cons,
    mult_decisions::MultiplierDecisionStore,
    key::Tuple{Int, Int, Int, Int};
    label=""
)
    m = length(domain_cons)
    mu = @variable(model, [1:m], lower_bound = 0.0)

    vars_p2 = [x1, x2, y1, y2]
    s_dec = new_sos_multiplier_decision!(
        model,
        vars_p2,
        ALT_P2_PREMISE_MULT_DEG;
        basename="p2mult"
    )
    mult_decisions.p2[key] = s_dec

    certificate_poly = conclusion_poly_fixed - s_dec.poly * premise_poly_fixed

    for i in 1:m
        certificate_poly -= mu[i] * domain_cons[i]
    end

    @constraint(model, certificate_poly in SOSCone())
    return
end

function add_implication_sos_constraint_phaseB!(
    model,
    conclusion_poly,
    premise_poly,
    domain_cons,
    mult_fixed,
    key::Tuple{Int, Int, Int, Int};
    label=""
)
    m = length(domain_cons)
    mu = @variable(model, [1:m], lower_bound = 0.0)

    s_fixed = fixed_p2_multiplier_poly_or_default(mult_fixed, key)

    certificate_poly = conclusion_poly - s_fixed * premise_poly

    for i in 1:m
        certificate_poly -= mu[i] * domain_cons[i]
    end

    @constraint(model, certificate_poly in SOSCone())
    return
end

function add_two_premise_implication_sos_constraint_phaseA!(
    model,
    conclusion_poly_fixed,
    premise1_poly_fixed,
    premise2_poly_fixed,
    domain_cons,
    mult_decisions::MultiplierDecisionStore,
    key::Tuple{Int, Int, Int};
    label=""
)
    m = length(domain_cons)
    mu = @variable(model, [1:m], lower_bound = 0.0)

    vars_p3 = [x01, x02, y1, y2, z1, z2]

    s1_dec = new_sos_multiplier_decision!(
        model,
        vars_p3,
        ALT_P3_PREMISE_MULT_DEG;
        basename="p3amult"
    )
    s2_dec = new_sos_multiplier_decision!(
        model,
        vars_p3,
        ALT_P3_PREMISE_MULT_DEG;
        basename="p3bmult"
    )

    mult_decisions.p3a[key] = s1_dec
    mult_decisions.p3b[key] = s2_dec

    certificate_poly =
        conclusion_poly_fixed -
        s1_dec.poly * premise1_poly_fixed -
        s2_dec.poly * premise2_poly_fixed

    for i in 1:m
        certificate_poly -= mu[i] * domain_cons[i]
    end

    @constraint(model, certificate_poly in SOSCone())
    return
end

function add_two_premise_implication_sos_constraint_phaseB!(
    model,
    conclusion_poly,
    premise1_poly,
    premise2_poly,
    domain_cons,
    mult_fixed,
    key::Tuple{Int, Int, Int};
    label=""
)
    m = length(domain_cons)
    mu = @variable(model, [1:m], lower_bound = 0.0)

    s1_fixed = fixed_p3a_multiplier_poly_or_default(mult_fixed, key)
    s2_fixed = fixed_p3b_multiplier_poly_or_default(mult_fixed, key)

    certificate_poly =
        conclusion_poly -
        s1_fixed * premise1_poly -
        s2_fixed * premise2_poly

    for i in 1:m
        certificate_poly -= mu[i] * domain_cons[i]
    end

    @constraint(model, certificate_poly in SOSCone())
    return
end

# ---------------------------------------------------------------------
# Phase A: fixed C, solve SOS premise multipliers
# ---------------------------------------------------------------------

function build_phaseA_problem(model, C_coeff_fixed)
    C_fixed = Matrix{Any}(undef, length(nodes), length(nodes))

    for p in nodes
        for q in nodes
            C_fixed[p, q] = fixed_C_poly(C_coeff_fixed, p, q)
        end
    end

    mult_decisions = empty_multiplier_decision_store()
    num_sos_constraints = 0

    # P1. One-step closure: C[p,q](x,f_sigma(x)) >= 0 on X.
    for (p, sigma, q) in edges
        poly = C_x_fx(C_fixed[p, q], sigma)
        add_domain_sos_constraint!(model, poly, X_cons_x; label="one_step_phaseA")
        num_sos_constraints += 1
    end

    # P2. Transitive implication.
    XY_cons = vcat(X_cons_x, X_cons_y)

    for (p, sigma, q) in edges
        for r in nodes
            key = (p, sigma, q, r)
            premise = C_fx_y(C_fixed[q, r], sigma)
            conclusion = C_x_y(C_fixed[p, r])

            add_implication_sos_constraint_phaseA!(
                model,
                conclusion,
                premise,
                XY_cons,
                mult_decisions,
                key;
                label="transitive_phaseA"
            )

            num_sos_constraints += 1
        end
    end

    # P3. Well-foundedness/ranking decrease.
    X0_Xu_Xu_cons = vcat(X0_cons, Xu_cons_y, Xu_cons_z)

    for p in nodes
        for q in nodes
            for r in nodes
                key = (p, q, r)
                premise1 = C_x0_y(C_fixed[p, q])
                premise2 = C_y_z(C_fixed[q, r])
                conclusion = C_x0_y(C_fixed[p, q]) - WF_DEC - C_x0_z(C_fixed[p, r])

                add_two_premise_implication_sos_constraint_phaseA!(
                    model,
                    conclusion,
                    premise1,
                    premise2,
                    X0_Xu_Xu_cons,
                    mult_decisions,
                    key;
                    label="well_founded_phaseA"
                )

                num_sos_constraints += 1
            end
        end
    end

    @objective(model, Min, 0.0)
    return mult_decisions, num_sos_constraints
end

# ---------------------------------------------------------------------
# Phase B: fixed multipliers, solve C
# ---------------------------------------------------------------------

function build_phaseB_problem(model, mult_fixed)
    @variable(model, coeff[p in nodes, q in nodes, k in 1:nbasis])

    C_dec = Matrix{Any}(undef, length(nodes), length(nodes))

    for p in nodes
        for q in nodes
            C_dec[p, q] = sum(coeff[p, q, k] * basis_xy[k] for k in 1:nbasis)
        end
    end

    num_anchor_constraints, anchor_slack_sum = 
        add_soft_negative_anchor_constraints!(model, C_dec)

    println("Soft negative anchor constraints = ", num_anchor_constraints)

    num_sos_constraints = 0

    # P1. One-step closure.
    for (p, sigma, q) in edges
        poly = C_x_fx(C_dec[p, q], sigma)
        add_domain_sos_constraint!(model, poly, X_cons_x; label="one_step_phaseB")
        num_sos_constraints += 1
    end

    # P2. Transitive implication.
    XY_cons = vcat(X_cons_x, X_cons_y)

    for (p, sigma, q) in edges
        for r in nodes
            key = (p, sigma, q, r)
            premise = C_fx_y(C_dec[q, r], sigma)
            conclusion = C_x_y(C_dec[p, r])

            add_implication_sos_constraint_phaseB!(
                model,
                conclusion,
                premise,
                XY_cons,
                mult_fixed,
                key;
                label="transitive_phaseB"
            )

            num_sos_constraints += 1
        end
    end

    # P3. Well-foundedness/ranking decrease.
    X0_Xu_Xu_cons = vcat(X0_cons, Xu_cons_y, Xu_cons_z)

    for p in nodes
        for q in nodes
            for r in nodes
                key = (p, q, r)
                premise1 = C_x0_y(C_dec[p, q])
                premise2 = C_y_z(C_dec[q, r])
                conclusion = C_x0_y(C_dec[p, q]) - WF_DEC - C_x0_z(C_dec[p, r])

                add_two_premise_implication_sos_constraint_phaseB!(
                    model,
                    conclusion,
                    premise1,
                    premise2,
                    X0_Xu_Xu_cons,
                    mult_fixed,
                    key;
                    label="well_founded_phaseB"
                )

                num_sos_constraints += 1
            end
        end
    end

    @objective(model, Min, anchor_slack_sum)
    return coeff, C_dec, num_sos_constraints
end

# ---------------------------------------------------------------------
# Solver utilities and extraction
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

function print_solve_summary(model, solver_name::Symbol, phase_name::String)
    println("\nSolver = ", solver_name)
    println("Phase  = ", phase_name)
    println("Termination status = ", termination_status(model))
    println("Primal status      = ", primal_status(model))
    if has_values(model)
        println("Objective value    = ", safe_objective_value(model))
    else
        println("Objective value    = unavailable because no primal values were returned.")
    end
end

function solve_phaseA_with_solver(C_coeff_fixed, solver_name::Symbol)
    println("\nBuilding Phase A fixed-C multiplier model with ", uppercase(string(solver_name)), "...")
    model = make_model(solver_name)
    mult_decisions, num_sos_constraints = build_phaseA_problem(model, C_coeff_fixed)

    println("Number of SOS/domain constraints = ", num_sos_constraints)
    println("Expected count for full implication model = 20")
    println("P2 multiplier decisions = ", length(mult_decisions.p2))
    println("P3A multiplier decisions = ", length(mult_decisions.p3a))
    println("P3B multiplier decisions = ", length(mult_decisions.p3b))
    println("Starting Phase A optimization...")

    optimize!(model)
    print_solve_summary(model, solver_name, "Phase A")

    return model, mult_decisions
end

function solve_phaseB_with_solver(mult_fixed, solver_name::Symbol; seed_phase=false)
    println("\nBuilding Phase B fixed-multiplier C model with ", uppercase(string(solver_name)), "...")
    model = make_model(solver_name)
    coeff, C_dec, num_sos_constraints = build_phaseB_problem(model, mult_fixed)

    println("Number of SOS/domain constraints = ", num_sos_constraints)
    println("Expected count for full implication model = 20")

    if mult_fixed === nothing
        println("Fixed P2 multipliers = baseline scalar IMP_MULT_P2")
        println("Fixed P3A multipliers = baseline scalar IMP_MULT_P3_A")
        println("Fixed P3B multipliers = baseline scalar IMP_MULT_P3_B")
    else
        println("Fixed P2 multipliers = ", length(mult_fixed.p2))
        println("Fixed P3A multipliers = ", length(mult_fixed.p3a))
        println("Fixed P3B multipliers = ", length(mult_fixed.p3b))
    end
    
    println("Starting Phase B optimization...")

    optimize!(model)

    print_solve_summary(
        model,
        solver_name,
        seed_phase ? "Bootstrap Phase B" : "Phase B"
    )

    return model, coeff, C_dec
end

function solve_multipliers_given_C(C_coeff_fixed)
    for solver_name in [:mosek, :scs]
        try
            model, mult_decisions = solve_phaseA_with_solver(C_coeff_fixed, solver_name)

            if is_acceptable_alternating_iterate(model)
                if !acceptable_solution(model)
                    println("[WARN] Phase A with ", solver_name, " returned only an approximate iterate.")
                    println("       This is allowed for alternating, but not accepted as a final certificate.")
                end

                mult_fixed = extract_fixed_multipliers(mult_decisions)
                return mult_fixed, model, solver_name
            else
                println("[WARN] Phase A with ", solver_name, " did not return an acceptable alternating iterate.")
            end
        catch err
            println("[WARN] Phase A with ", solver_name, " failed.")
            println("Error:")
            println(err)
        end
    end

    return nothing, nothing, :none
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

function solve_C_given_multipliers(mult_fixed; seed_phase=false)
    for solver_name in [:mosek, :scs]
        try
            model, coeff, C_dec = solve_phaseB_with_solver(
                mult_fixed,
                solver_name;
                seed_phase=seed_phase
            )

            accepted =
                seed_phase ? is_acceptable_seed_solution(model) : is_acceptable_phaseB_iterate(model)

            if accepted
                if seed_phase && !acceptable_solution(model)
                    println("[WARN] Bootstrap Phase B with ", solver_name, " returned only an approximate seed.")
                    println("       This is allowed for initialization, but not accepted as a final certificate.")
                elseif !seed_phase && !acceptable_solution(model)
                    println("[WARN] Phase B with ", solver_name, " returned only an approximate C iterate.")
                    println("       This is allowed for alternating, but not accepted as a final certificate.")
                end

                C_coeff_new = extract_C_coeffs(coeff)
                return C_coeff_new, model, solver_name
            else
                if seed_phase
                    println("[WARN] Bootstrap Phase B with ", solver_name, " did not return an acceptable seed.")
                else
                    println("[WARN] Phase B with ", solver_name, " did not return an acceptable alternating iterate.")
                end
            end
        catch err
            if seed_phase
                println("[WARN] Bootstrap Phase B with ", solver_name, " failed.")
            else
                println("[WARN] Phase B with ", solver_name, " failed.")
            end
            println("Error:")
            println(err)
        end
    end

    return nothing, nothing, :none
end

# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

function save_status_only(status_dict)
    mkpath(RESULTS_DIR)
    open(OUT_STATUS_JSON, "w") do io
        JSON.print(io, status_dict, 2)
    end
    println("\nSaved alternating status summary:")
    println("  ", OUT_STATUS_JSON)
end

function serialize_fixed_multiplier_store(mult_fixed)
    if mult_fixed === nothing
        return nothing
    end

    p2 = Dict{String, Any}()
    p3a = Dict{String, Any}()
    p3b = Dict{String, Any}()

    for (key, mult) in mult_fixed.p2
        p2[string(key)] = Dict(
            "coeffs" => mult.coeffs,
            "basis" => [string(b) for b in mult.basis],
        )
    end

    for (key, mult) in mult_fixed.p3a
        p3a[string(key)] = Dict(
            "coeffs" => mult.coeffs,
            "basis" => [string(b) for b in mult.basis],
        )
    end

    for (key, mult) in mult_fixed.p3b
        p3b[string(key)] = Dict(
            "coeffs" => mult.coeffs,
            "basis" => [string(b) for b in mult.basis],
        )
    end

    return Dict(
        "p2" => p2,
        "p3a" => p3a,
        "p3b" => p3b,
    )
end

function save_alternating_results(
    C_coeff,
    best_mult,
    last_phaseB_model,
    phaseA_solver,
    phaseB_solver,
    iterations_completed
)
    mkpath(RESULTS_DIR)

    coeff_values = coeff_dict_to_array(C_coeff)
    basis_strings = [string(b) for b in basis_xy]
    edge_array = [[p, sigma, q] for (p, sigma, q) in edges]

    status_string =
        last_phaseB_model === nothing ? "UNKNOWN" : string(termination_status(last_phaseB_model))

    primal_status_string =
        last_phaseB_model === nothing ? "UNKNOWN" : string(primal_status(last_phaseB_model))

    objective =
        last_phaseB_model === nothing ? nothing : safe_objective_value(last_phaseB_model)

    final_solver_accepted =
        last_phaseB_model !== nothing && acceptable_solution(last_phaseB_model)

    certificate_type = "alternating_implication_pairwise_sos"

    serialized_multipliers = serialize_fixed_multiplier_store(best_mult)

    @save OUT_JLD2 coeff_values basis_strings edge_array WF_DEC status_string primal_status_string objective certificate_type iterations_completed serialized_multipliers

    json_dict = Dict(
        "description" => "Alternating implication-style SOS PC-CC synthesis result for two-car platoon example",
        "certificate_type" => certificate_type,
        "phaseA_solver" => string(phaseA_solver),
        "phaseB_solver" => string(phaseB_solver),
        "status" => status_string,
        "primal_status" => primal_status_string,
        "objective" => objective,
        "iterations_completed" => iterations_completed,
        "feasible_certificate_saved" => final_solver_accepted,
        "candidate_saved" => true,
        "paper_ready_solver_status" => final_solver_accepted,
        "certificate_warning" => final_solver_accepted ?
            "Final Phase B solver status was accepted." :
            "Candidate saved from an approximate alternating iterate; validate carefully and do not use as a paper certificate without further verification.",
        "wf_decrease" => WF_DEC,
        "C_degree" => 2,
        "P2_premise_multiplier_degree" => ALT_P2_PREMISE_MULT_DEG,
        "P3_premise_multiplier_degree" => ALT_P3_PREMISE_MULT_DEG,
        "fixed_premise_multipliers" => serialized_multipliers,
        "baseline_fixed_multipliers" => Dict(
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
            "P2" => "For each edge (p,sigma,q) and node r, C[q,r](f_sigma(x),y) >= 0 implies C[p,r](x,y) >= 0 on X x X.",
            "P3" => "For all p,q,r, C[p,q](x0,y) >= 0 and C[q,r](y,z) >= 0 imply C[p,r](x0,z) <= C[p,q](x0,y) - wf_decrease on X0 x Xu x Xu.",
        ),
    )

    open(OUT_JSON, "w") do io
        JSON.print(io, json_dict, 2)
    end

    println("\nSaved alternating SOS results:")
    println("  ", OUT_JLD2)
    println("  ", OUT_JSON)
end

# ---------------------------------------------------------------------
# Main alternating loop
# ---------------------------------------------------------------------

function main()
    println()
    println("============================================================")
    println("Starting alternating SOS synthesis")
    println("============================================================")

    C_coeff_current, seed_model, seed_solver = solve_C_given_multipliers(nothing; seed_phase=true)

    if C_coeff_current === nothing
        println("\n[ERROR] Bootstrap Phase B failed.")
        println("        Could not obtain an initial C from fixed scalar multipliers.")
        println("        Alternating synthesis cannot start.")

        status_dict = Dict(
            "description" => "Alternating implication-style SOS PC-CC synthesis status for two-car platoon example",
            "certificate_type" => "alternating_implication_pairwise_sos",
            "feasible_certificate_saved" => false,
            "message" => "Bootstrap Phase B failed; no initial C seed was obtained.",
            "iterations_completed" => 0,
            "bootstrap_solver" => string(seed_solver),
            "wf_decrease" => WF_DEC,
            "C_degree" => 2,
            "P2_premise_multiplier_degree" => ALT_P2_PREMISE_MULT_DEG,
            "P3_premise_multiplier_degree" => ALT_P3_PREMISE_MULT_DEG,
            "nodes" => nodes,
            "edges" => [[p, sigma, q] for (p, sigma, q) in edges],
        )

        save_status_only(status_dict)

        println("\nDone, but no alternating certificate was saved.")
        return
    end

    println("\n[OK] Bootstrap Phase B produced an initial C seed.")
    println("     bootstrap solver = ", seed_solver)
    println("     bootstrap termination status = ", termination_status(seed_model))
    println("     bootstrap primal status      = ", primal_status(seed_model))

    best_C_coeff = nothing
    best_mult = nothing
    last_phaseB_model = nothing
    last_phaseA_solver = :none
    last_phaseB_solver = :none
    iterations_completed = 0

    for iter in 1:ALT_MAX_ITERS
        println()
        println("============================================================")
        println("Alternating iteration ", iter, " / ", ALT_MAX_ITERS)
        println("============================================================")

        println("\n=== Phase A: fixed C, solve SOS premise multipliers ===")
        mult_fixed, phaseA_model, phaseA_solver = solve_multipliers_given_C(C_coeff_current)

        if mult_fixed === nothing
            println("\n[WARN] Phase A failed at iteration ", iter, ".")
            println("       No multiplier set was extracted.")
            break
        end

        println("\n[OK] Phase A succeeded with solver ", phaseA_solver, ".")
        println("Extracted P2 multipliers  = ", length(mult_fixed.p2))
        println("Extracted P3A multipliers = ", length(mult_fixed.p3a))
        println("Extracted P3B multipliers = ", length(mult_fixed.p3b))

        println("\n=== Phase B: fixed multipliers, solve C ===")
        C_coeff_new, phaseB_model, phaseB_solver = solve_C_given_multipliers(mult_fixed)

        if C_coeff_new === nothing
            println("\n[WARN] Phase B failed at iteration ", iter, ".")
            println("       No new C coefficient set was extracted.")
            break
        end

        delta_C = max_C_coeff_change(C_coeff_current, C_coeff_new)
        println("\n[OK] Phase B succeeded with solver ", phaseB_solver, ".")
        println("Max C coefficient change = ", delta_C)

        best_C_coeff = C_coeff_new
        best_mult = mult_fixed
        last_phaseB_model = phaseB_model
        last_phaseA_solver = phaseA_solver
        last_phaseB_solver = phaseB_solver
        iterations_completed = iter
        C_coeff_current = C_coeff_new

        if delta_C < ALT_C_CONV_TOL
            println("\n[OK] Alternating scheme converged by coefficient change.")
            break
        end
    end

    println()
    println("============================================================")
    println("Alternating loop finished")
    println("============================================================")
    println("Iterations completed = ", iterations_completed)

    if best_C_coeff === nothing
        println("\n[WARN] No complete Phase A/Phase B iteration succeeded.")
        status_dict = Dict(
            "description" => "Alternating implication-style SOS PC-CC synthesis status for two-car platoon example",
            "certificate_type" => "alternating_implication_pairwise_sos",
            "feasible_certificate_saved" => false,
            "message" => "No result was saved because no complete Phase A/Phase B iteration succeeded.",
            "iterations_completed" => iterations_completed,
            "wf_decrease" => WF_DEC,
            "C_degree" => 2,
            "P2_premise_multiplier_degree" => ALT_P2_PREMISE_MULT_DEG,
            "P3_premise_multiplier_degree" => ALT_P3_PREMISE_MULT_DEG,
            "nodes" => nodes,
            "edges" => [[p, sigma, q] for (p, sigma, q) in edges],
        )
        save_status_only(status_dict)
        println("\nDone, but no alternating certificate was saved.")
        return
    end

    println("\nSaving best C coefficient set from alternating synthesis...")
    
    save_alternating_results(
        best_C_coeff,
        best_mult,
        last_phaseB_model,
        last_phaseA_solver,
        last_phaseB_solver,
        iterations_completed
    )

    println("\nDone.")
end

main()
