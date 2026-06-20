# Export a deterministic polynomial PC-CC certificate for the nonlinear
# two-car platoon case study on the one-node path-complete graph.
#
# Graph:
#   one node v1 with self-loops labeled 1 and 2.
#
# Certificate:
#   C[1,1](x,y) = (y2 - y1) - 0.40001.
#
# This is the same low-degree polynomial certificate used for the two-node
# graph. It lies in the linear subspace of the quadratic SOS/TSSOS-compatible
# template. Since it is independent of graph-node indices, it also verifies the
# one-node common-closure graph.
#
# Run from repo root:
#   julia --project=. sos/pccc_platoon_export_polynomial_certificate_1node.jl
#
# Validate with the existing persistence-aware validator:
#   julia --project=. sos/pccc_validate_platoon_persistence.jl results/pccc_platoon_polynomial_certificate_1node.json 21 41

using JSON

const RESULTS_DIR = "results"
const OUT_JSON = joinpath(RESULTS_DIR, "pccc_platoon_polynomial_certificate_1node.json")

const nodes = [1]
const edges = [[1, 1, 1], [1, 2, 1]]
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
# coeff_values[k][q][p]. For the one-node graph this is just a 1x1 block.
function coeff_block(v)
    return [[v]]
end

coeffs = Float64[]
push!(coeffs, -(GAP_THRESHOLD + STRICT_SHIFT)) # constant term
push!(coeffs, 1.0)                             # y2
push!(coeffs, -1.0)                            # y1
for _ in 4:length(basis_xy)
    push!(coeffs, 0.0)
end
coeff_values = [coeff_block(c) for c in coeffs]

json_dict = Dict(
    "description" => "Explicit low-degree polynomial PC-CC certificate for the nonlinear two-car platoon example on the one-node path-complete graph.",
    "certificate_type" => "analytic_polynomial_gap_certificate_one_node",
    "status" => "EXPLICIT_CERTIFICATE",
    "primal_status" => "EXPLICIT_CERTIFICATE",
    "objective" => nothing,
    "iterations_completed" => 0,
    "feasible_certificate_saved" => true,
    "candidate_saved" => false,
    "paper_ready_solver_status" => true,
    "certificate_warning" => "Explicit certificate represented within the quadratic SOS/TSSOS-compatible template. The active P3 antecedent is empty on X0 x Xvf x Xvf.",
    "wf_decrease" => WF_DEC,
    "C_degree" => 1,
    "C_template_degree_limit" => 2,
    "nodes" => nodes,
    "edges" => edges,
    "basis_xy" => basis_xy,
    "coeff_values" => coeff_values,
    "state_domain" => Dict("x1_min" => 0.0, "x1_max" => 3.0, "x2_min" => 0.0, "x2_max" => 5.1, "gap_min" => 0.0),
    "initial_set" => Dict("gap_max" => 0.3),
    "persistence_set" => Dict("gap_max" => GAP_THRESHOLD, "interpretation" => "Xvf is the finite-visit/small-gap region."),
    "unsafe_set" => Dict("gap_max" => GAP_THRESHOLD, "note" => "Kept for backward compatibility with existing scripts; interpret as finite-visit region in the paper."),
    "explicit_polynomial" => "C[1,1](x,y) = y2 - y1 - 0.40001.",
    "graph_interpretation" => "One-node path-complete graph with self-loops for modes 1 and 2; this is the common closure-certificate case.",
    "conditions" => Dict(
        "P1" => "C[1,1](x,f_sigma(x)) >= 0 because f_sigma(X) is separated from Xvf by a large gap margin.",
        "P2" => "C depends only on terminal state y; hence active premise C(f_sigma(x),y)>=0 implies the identical conclusion C(x,y)>=0.",
        "P3" => "For y,z in Xvf, C(x0,y) and C(y,z) are strictly negative; hence the antecedent of P3 is false and the implication holds."
    )
)

mkpath(RESULTS_DIR)
open(OUT_JSON, "w") do io
    JSON.print(io, json_dict, 2)
end
println("Saved explicit one-node polynomial PC-CC certificate:")
println("  ", OUT_JSON)
println("Validate with:")
println("  julia --project=. sos/pccc_validate_platoon_persistence.jl ", OUT_JSON, " 21 41")
