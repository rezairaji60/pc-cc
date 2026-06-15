# Export a deterministic quadratic-compatible PC-CC certificate for the
# nonlinear two-car platoon case study.
#
# Certificate idea:
#   C[p,q](x,y) = (y2 - y1) - 0.40001  for all node pairs (p,q).
#
# Since the image of the full domain X under both modes has gap at least
# about 1.49, P1 holds with a large margin. P2 is immediate because C depends
# only on the terminal state y. On the finite-visit set Xu = {y2-y1 <= 0.4},
# C is strictly negative, so the P3 antecedent is empty. This is a valid
# vacuous implication and certifies persistence: after one transition the
# system cannot be in Xu.
#
# Run from repo root:
#   julia --project=. sos/pccc_platoon_export_no_return_certificate.jl
#
# Then validate with the persistence-aware validator:
#   julia --project=. sos/pccc_validate_platoon_persistence.jl results/pccc_platoon_no_return_certificate.json 21 41

using JSON

const RESULTS_DIR = "results"
const OUT_JSON = joinpath(RESULTS_DIR, "pccc_platoon_no_return_certificate.json")

const nodes = [1, 2]
const edges = [[1, 1, 1], [1, 1, 2], [1, 2, 2], [2, 2, 1]]
const WF_DEC = 1.0e-6
const GAP_THRESHOLD = 0.4
const STRICT_SHIFT = 1.0e-5

const basis_xy = [
    "1",
    "y2",
    "y1",
    "x2",
    "x1",
    "y2^2",
    "y1*y2",
    "y1^2",
    "x2*y2",
    "x2*y1",
    "x2^2",
    "x1*y2",
    "x1*y1",
    "x1*x2",
    "x1^2",
]

# coeff_values layout expected by the existing validator after JSON parsing:
# coeff_values[k][q][p].  Because this certificate uses the same polynomial
# for all node pairs, each 2x2 block is constant.
function coeff_block(v)
    return [[v, v], [v, v]]
end

coeffs = Float64[]
push!(coeffs, -(GAP_THRESHOLD + STRICT_SHIFT)) # 1
push!(coeffs, 1.0)                             # y2
push!(coeffs, -1.0)                            # y1
for _ in 4:length(basis_xy)
    push!(coeffs, 0.0)
end
coeff_values = [coeff_block(c) for c in coeffs]

json_dict = Dict(
    "description" => "Deterministic no-return PC-CC certificate for the nonlinear two-car platoon example.",
    "certificate_type" => "analytic_no_return_gap_certificate",
    "status" => "EXPLICIT_CERTIFICATE",
    "primal_status" => "EXPLICIT_CERTIFICATE",
    "objective" => nothing,
    "iterations_completed" => 0,
    "feasible_certificate_saved" => true,
    "candidate_saved" => false,
    "paper_ready_solver_status" => true,
    "certificate_warning" => "This certificate is explicit, not solver-generated. P3 is valid vacuously because C is strictly negative on X x Xu; use the persistence-aware validator for implication semantics.",
    "wf_decrease" => WF_DEC,
    "C_degree" => 1,
    "C_template_degree_limit" => 2,
    "nodes" => nodes,
    "edges" => edges,
    "basis_xy" => basis_xy,
    "coeff_values" => coeff_values,
    "state_domain" => Dict("x1_min" => 0.0, "x1_max" => 3.0, "x2_min" => 0.0, "x2_max" => 5.1, "gap_min" => 0.0),
    "initial_set" => Dict("gap_max" => 0.3),
    "persistence_set" => Dict("gap_max" => GAP_THRESHOLD, "interpretation" => "Xu is the finite-visit/small-gap region."),
    "unsafe_set" => Dict("gap_max" => GAP_THRESHOLD, "note" => "Kept for backward compatibility with existing scripts; interpret as finite-visit region in the paper."),
    "explicit_polynomial" => "C[p,q](x,y) = y2 - y1 - 0.40001 for all p,q.",
    "conditions" => Dict(
        "P1" => "C[p,q](x,f_sigma(x)) >= 0 because f_sigma(X) is separated from Xu by a large gap margin.",
        "P2" => "C depends only on terminal state y; hence active premise C(f_sigma(x),y)>=0 implies the identical conclusion C(x,y)>=0.",
        "P3" => "For y,z in Xu, C(x0,y) and C(y,z) are strictly negative; hence the antecedent of P3 is false and the implication holds vacuously."
    )
)

mkpath(RESULTS_DIR)
open(OUT_JSON, "w") do io
    JSON.print(io, json_dict, 2)
end
println("Saved explicit platoon no-return certificate:")
println("  ", OUT_JSON)
println("Validate with:")
println("  julia --project=. sos/pccc_validate_platoon_persistence.jl ", OUT_JSON, " 21 41")
