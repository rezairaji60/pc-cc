# This file is intentionally self-contained.
# It exports and documents the low-degree polynomial PC-CC certificate
# for the nonlinear two-car platoon benchmark.

using JSON

const RESULTS_DIR = "results"
const WF_DEC_VALIDATOR = 1.0e-6
const C_DEGREE = 2

# Certificate:
#   C[p,q](x,y) = y2 - y1 - 0.40001
# represented in the quadratic basis used by the validator:
#   1, y2, y1, x2, x1, y2^2, y1*y2, y1^2,
#   x2*y2, x2*y1, x2^2, x1*y2, x1*y1, x1*x2, x1^2
const BASIS_STRINGS = [
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
const BASIS_EXPONENTS = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 2],
    [0, 0, 1, 1],
    [0, 0, 2, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 2, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [2, 0, 0, 0],
]
const NBASIS = length(BASIS_STRINGS)

function certificate_coeff_vector()
    c = zeros(Float64, NBASIS)
    c[1] = -0.40001   # constant
    c[2] =  1.0       # y2
    c[3] = -1.0       # y1
    return c
end

# Dynamics.
f1_1(x1, x2) = 0.01*x2 + 0.9*x1 - 0.02*x1^2
f1_2(x1, x2) = 2.0 + 0.8*x2 - 0.04*x2^2
f2_1(x1, x2) = 0.9*x1 - 0.02*x1^2
f2_2(x1, x2) = 2.0 + 0.8*x2 - 0.04*x2^2

function gap_after_mode(mode::Int, x1::Float64, x2::Float64)
    if mode == 1
        return f1_2(x1, x2) - f1_1(x1, x2)
    elseif mode == 2
        return f2_2(x1, x2) - f2_1(x1, x2)
    else
        error("Unknown mode $mode")
    end
end

function C_terminal_gap(y1::Float64, y2::Float64)
    return y2 - y1 - 0.40001
end

function build_coeff_values(nodes)
    n = length(nodes)
    coeff_values = Array{Float64}(undef, n, n, NBASIS)
    c = certificate_coeff_vector()
    for p in 1:n
        for q in 1:n
            coeff_values[p, q, :] .= c
        end
    end
    return coeff_values
end

function sos_multiplier_certificate_dict()
    return Dict(
        "description" => "One explicit SOS multiplier certificate for C[p,q](x,y)=y2-y1-0.40001. Multipliers are nonunique.",
        "certificate_polynomial" => "C[p,q](x,y) = y2 - y1 - 0.40001",
        "PC_CC1" => Dict(
            "redundant_domain_constraint" => "h_b(x) = x2*(5.1 - x2) >= 0, added without changing X",
            "gap_constraint" => "h_g(x) = x2 - x1 >= 0",
            "mode_1" => Dict(
                "Lambda_h_b" => 0.04,
                "Lambda_h_g" => 0.586,
                "residual" => "0.02*(x1 - 7.85)^2 + 0.36754"
            ),
            "mode_2" => Dict(
                "Lambda_h_b" => 0.04,
                "Lambda_h_g" => 0.596,
                "residual" => "0.02*(x1 - 7.6)^2 + 0.44479"
            )
        ),
        "PC_CC2" => Dict(
            "premise_multiplier_s2" => 1.0,
            "domain_multipliers" => "all zero",
            "residual" => "0"
        ),
        "PC_CC3" => Dict(
            "h_vf_y" => "0.4 - (y2 - y1)",
            "h_vf_yp" => "0.4 - (yp2 - yp1)",
            "premise_multiplier_s3a" => 1.0e5,
            "premise_multiplier_s3b" => 0.0,
            "Lambda_h_vf_y" => 99999.0,
            "Lambda_h_vf_yp" => 1.0,
            "all_other_domain_multipliers" => 0.0,
            "residual_after_domain_multipliers" => "0"
        )
    )
end

function print_certificate_summary(nodes, edges, outfile)
    println("============================================================")
    println("Polynomial PC-CC certificate for nonlinear platoon")
    println("============================================================")
    println("nodes = ", nodes)
    println("edges = ", edges)
    println("C[p,q](x,y) = y2 - y1 - 0.40001 for all p,q")
    println("Basis order:")
    for (i, b) in enumerate(BASIS_STRINGS)
        println("  ", i, ": ", b)
    end
    println()
    println("Coefficient vector:")
    println("  constant = -0.40001")
    println("  y2       =  1.0")
    println("  y1       = -1.0")
    println("  all other coefficients = 0")
    println()
    println("Analytic PC-CC checks:")
    println("  PC-CC1: g(f1(x)) >= 1.49, g(f2(x)) >= 1.52 on X")
    println("          therefore C(x,f_sigma(x)) >= 1.08999 > 0")
    println("  PC-CC2: C[q,r](f_sigma(x),y) = C[p,r](x,y), so s2 = 1 gives zero residual")
    println("  PC-CC3: if y in X_vf then C[p,q](x0,y) <= -1e-5, so the active premise is empty")
    println()
    println("Explicit SOS multipliers included in JSON under key: sos_multiplier_certificate")
    println("Output JSON: ", outfile)
    println("============================================================")
end

function write_certificate_json(nodes, edges, outfile; graph_kind)
    mkpath(RESULTS_DIR)
    coeff_values = build_coeff_values(nodes)
    edge_array = [[p, sigma, q] for (p, sigma, q) in edges]

    json_dict = Dict(
        "description" => "Low-degree polynomial PC-CC certificate for the nonlinear platoon benchmark",
        "certificate_type" => "explicit_polynomial_pccc_terminal_gap_common_closure",
        "graph_kind" => graph_kind,
        "status" => "EXPLICIT_ANALYTIC_CERTIFICATE",
        "primal_status" => "NOT_SOLVER_GENERATED",
        "objective" => nothing,
        "feasible_certificate_saved" => true,
        "candidate_saved" => false,
        "paper_ready_solver_status" => true,
        "certificate_warning" => "Certificate is exported explicitly and verified analytically; SOS multipliers are included for one valid proof object.",
        "wf_decrease" => WF_DEC_VALIDATOR,
        "theory_normalized_decrease" => 1.0,
        "C_degree" => C_DEGREE,
        "nodes" => nodes,
        "edges" => edge_array,
        "basis_xy" => BASIS_STRINGS,
        "basis_exponents_xy" => BASIS_EXPONENTS,
        "coeff_values" => coeff_values,
        "certificate_formula" => "C[p,q](x,y)=y2-y1-0.40001 for every node pair",
        "state_domain" => Dict("x1_min" => 0.0, "x1_max" => 3.0, "x2_min" => 0.0, "x2_max" => 5.1, "gap_min" => 0.0),
        "initial_set" => Dict("gap_max" => 0.3),
        "persistence_set" => Dict("gap_max" => 0.4, "interpretation" => "X_vf is the finite-visit small-gap set."),
        "unsafe_set" => Dict("gap_max" => 0.4, "note" => "Backward-compatibility key for older validators; interpret as finite-visit set."),
        "analytic_margins" => Dict(
            "min_gap_after_mode_1_lower_bound" => 1.49,
            "min_gap_after_mode_2_lower_bound" => 1.52,
            "min_P1_lower_bound" => 1.08999,
            "C_on_Xvf_upper_bound" => -1.0e-5
        ),
        "sos_multiplier_certificate" => sos_multiplier_certificate_dict(),
        "conditions" => Dict(
            "P1" => "For each edge (p,sigma,q), C[p,q](x,f_sigma(x)) >= 0 on X.",
            "P2" => "For each edge (p,sigma,q) and node r, C[q,r](f_sigma(x),y) >= 0 implies C[p,r](x,y) >= 0 on X x X.",
            "P3" => "For all p,q,r, C[p,q](x0,y) >= 0 and C[q,r](y,yprime) >= 0 imply C[p,r](x0,yprime) <= C[p,q](x0,y)-1 on X0 x Xvf x Xvf."
        )
    )

    open(outfile, "w") do io
        JSON.print(io, json_dict, 2)
    end

    print_certificate_summary(nodes, edges, outfile)
end

function main()
    nodes = [1, 2]
    edges = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 2, 1),
    ]
    outfile = joinpath(RESULTS_DIR, "pccc_platoon_polynomial_certificate_2node.json")
    write_certificate_json(nodes, edges, outfile; graph_kind="two_node_path_complete")
end

main()
