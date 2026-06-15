# Direct dense-grid validation for the platoon PC-CC SOS certificate.
#
# Run from the repository root:
#
#   julia --project=. sos/pccc_validate_platoon_sos.jl
#
# Optional arguments:
#
#   julia --project=. sos/pccc_validate_platoon_sos.jl results/pccc_synth_platoon_sos_A_wf1e-5.json 21 41
#
# The script validates the actual polynomial values saved in the JSON file,
# not the SOS Gram matrices.

using JSON
using Printf

# ---------------------------------------------------------------------
# User validation parameters
# ---------------------------------------------------------------------

const DEFAULT_CERT_PATHS = [
    joinpath("results", "pccc_synth_platoon_sos_alternating.json"),
    joinpath("results", "pccc_synth_platoon_sos.json"),
    joinpath("results", "pccc_synth_platoon_sos_A_wf1e-5.json"),
    joinpath("results", "pccc_tuning_logs", "A_wf1e-5_certificate.json"),
]

const DEFAULT_N1 = 21
const DEFAULT_N2 = 41
const DEFAULT_TOL = 1.0e-7

# ---------------------------------------------------------------------
# Dynamics and graph
# ---------------------------------------------------------------------

function f_sigma(sigma::Int, a1::Float64, a2::Float64)
    if sigma == 1
        return 0.01 * a2 + 0.9 * a1 - 0.02 * a1^2,
               2.0 + 0.8 * a2 - 0.04 * a2^2
    elseif sigma == 2
        return 0.9 * a1 - 0.02 * a1^2,
               2.0 + 0.8 * a2 - 0.04 * a2^2
    else
        error("Unknown mode sigma = $sigma")
    end
end

function make_X_grid(N1::Int, N2::Int; gap_max=nothing)
    pts = Vector{NTuple{2, Float64}}()
    for x1 in range(0.0, 3.0; length=N1)
        for x2 in range(0.0, 5.1; length=N2)
            gap = x2 - x1
            if gap >= -1.0e-12
                if gap_max === nothing || gap <= gap_max + 1.0e-12
                    push!(pts, (Float64(x1), Float64(x2)))
                end
            end
        end
    end
    return pts
end

# ---------------------------------------------------------------------
# Certificate loading and monomial evaluation
# ---------------------------------------------------------------------

function resolve_cert_path(args)
    if length(args) >= 1
        path = args[1]
        if !isfile(path)
            error("Certificate JSON file not found: $path")
        end
        return path
    end

    for path in DEFAULT_CERT_PATHS
        if isfile(path)
            return path
        end
    end

    error("No certificate JSON file found. Tried:\n  " * join(DEFAULT_CERT_PATHS, "\n  "))
end

function normalize_monomial_string(s::AbstractString)
    t = replace(String(s), " " => "")
    t = replace(t, "·" => "*")
    # Convert common unicode superscripts to ^digits.
    t = replace(t, "²" => "^2")
    t = replace(t, "³" => "^3")
    t = replace(t, "⁴" => "^4")
    t = replace(t, "⁵" => "^5")
    t = replace(t, "⁶" => "^6")
    t = replace(t, "⁷" => "^7")
    t = replace(t, "⁸" => "^8")
    t = replace(t, "⁹" => "^9")
    return t
end

function monomial_exponents(s::AbstractString)
    t = normalize_monomial_string(s)
    exps = Dict("x1" => 0, "x2" => 0, "y1" => 0, "y2" => 0)

    if t == "1" || isempty(t)
        return (0, 0, 0, 0)
    end

    # Accept forms such as x1, x1^2, x1*x2, x1^2*y2, etc.
    for m in eachmatch(r"(x1|x2|y1|y2)(?:\^([0-9]+))?", t)
        var = m.captures[1]
        pow = m.captures[2] === nothing ? 1 : parse(Int, m.captures[2])
        exps[var] += pow
    end

    return (exps["x1"], exps["x2"], exps["y1"], exps["y2"])
end

function basis_values(exponents, a1::Float64, a2::Float64, b1::Float64, b2::Float64)
    vals = Vector{Float64}(undef, length(exponents))
    @inbounds for k in eachindex(exponents)
        e1, e2, e3, e4 = exponents[k]
        vals[k] = (a1^e1) * (a2^e2) * (b1^e3) * (b2^e4)
    end
    return vals
end

function coeff_at(coeff_values, p::Int, q::Int, k::Int, nbasis::Int)
    # The synthesis script stores coefficients as coeff_values[p,q,k].
    # After JSON.jl saves a 3-D Julia array and JSON.parsefile reads it back,
    # the nested JSON layout is coeff_values[k][q][p].
    if length(coeff_values) == nbasis
        return Float64(coeff_values[k][q][p])
    end

    # Fallback for manually nested layout coeff_values[p][q][k].
    return Float64(coeff_values[p][q][k])
end

function C_value(coeff_values, exponents, p::Int, q::Int,
                 a1::Float64, a2::Float64, b1::Float64, b2::Float64)
    vals = basis_values(exponents, a1, a2, b1, b2)
    nbasis = length(vals)
    s = 0.0

    @inbounds for k in eachindex(vals)
        s += coeff_at(coeff_values, p, q, k, nbasis) * vals[k]
    end

    return s
end

function get_edges(data)
    return [(Int(e[1]), Int(e[2]), Int(e[3])) for e in data["edges"]]
end

# ---------------------------------------------------------------------
# Validation routines
# ---------------------------------------------------------------------

function validate_P1(coeff_values, exponents, edges, X)
    minval = Inf
    worst = nothing

    for (p, sigma, q) in edges
        for x in X
            fx1, fx2 = f_sigma(sigma, x[1], x[2])
            val = C_value(coeff_values, exponents, p, q, x[1], x[2], fx1, fx2)
            if val < minval
                minval = val
                worst = (edge=(p, sigma, q), x=x, fx=(fx1, fx2))
            end
        end
    end

    return minval, worst
end

function validate_P2(coeff_values, exponents, edges, nodes, X, tol)
    active_count = 0
    min_active_B = Inf
    worst = nothing

    for (p, sigma, q) in edges
        for r in nodes
            for x in X
                fx1, fx2 = f_sigma(sigma, x[1], x[2])
                for y in X
                    A = C_value(coeff_values, exponents, q, r, fx1, fx2, y[1], y[2])
                    if A >= -tol
                        active_count += 1
                        B = C_value(coeff_values, exponents, p, r, x[1], x[2], y[1], y[2])
                        if B < min_active_B
                            min_active_B = B
                            worst = (edge=(p, sigma, q), r=r, x=x, y=y, A=A, B=B)
                        end
                    end
                end
            end
        end
    end

    return active_count, min_active_B, worst
end

function validate_P3(coeff_values, exponents, nodes, X0, Xu, xi, tol)
    active_count = 0
    exact_active_count = 0

    min_uncond_D = Inf
    min_active_D = Inf
    min_exact_active_D = Inf

    worst_uncond = nothing
    worst_active = nothing
    worst_exact_active = nothing

    for p in nodes
        for q in nodes
            for r in nodes
                for x0 in X0
                    for y in Xu
                        A = C_value(coeff_values, exponents, p, q, x0[1], x0[2], y[1], y[2])
                        for z in Xu
                            B = C_value(coeff_values, exponents, q, r, y[1], y[2], z[1], z[2])
                            Cpr = C_value(coeff_values, exponents, p, r, x0[1], x0[2], z[1], z[2])
                            D = A - xi - Cpr

                            if D < min_uncond_D
                                min_uncond_D = D
                                worst_uncond = (p=p, q=q, r=r, x0=x0, y=y, z=z, A=A, B=B, D=D)
                            end

                            # Conservative numerical activation:
                            # treats near-boundary premises as active.
                            if A >= -tol && B >= -tol
                                active_count += 1
                                if D < min_active_D
                                    min_active_D = D
                                    worst_active = (p=p, q=q, r=r, x0=x0, y=y, z=z, A=A, B=B, D=D)
                                end
                            end

                            # Exact mathematical activation:
                            # checks only points where both premises are actually nonnegative.
                            if A >= 0.0 && B >= 0.0
                                exact_active_count += 1
                                if D < min_exact_active_D
                                    min_exact_active_D = D
                                    worst_exact_active = (p=p, q=q, r=r, x0=x0, y=y, z=z, A=A, B=B, D=D)
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return active_count,
           exact_active_count,
           min_uncond_D,
           min_active_D,
           min_exact_active_D,
           worst_uncond,
           worst_active,
           worst_exact_active
end

function pass_fail(value, tol)
    return value >= -tol ? "PASS" : "FAIL"
end

function print_worst(label, worst)
    println(label)
    if worst === nothing
        println("  none")
    else
        for name in fieldnames(typeof(worst))
            println("  ", name, " = ", getfield(worst, name))
        end
    end
end

function main()
    cert_path = resolve_cert_path(ARGS)
    N1 = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : DEFAULT_N1
    N2 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : DEFAULT_N2
    tol = DEFAULT_TOL

    println("Loading certificate: ", cert_path)
    data = JSON.parsefile(cert_path)

    coeff_values = data["coeff_values"]
    basis_strings = data["basis_xy"]
    exponents = [monomial_exponents(s) for s in basis_strings]
    nodes = [Int(v) for v in data["nodes"]]
    edges = get_edges(data)
    xi = Float64(data["wf_decrease"])

    X = make_X_grid(N1, N2)
    X0 = make_X_grid(N1, N2; gap_max=0.3)
    Xu = make_X_grid(N1, N2; gap_max=0.4)

    println("\nDense-grid validation settings")
    println("  N1 = ", N1)
    println("  N2 = ", N2)
    println("  tol = ", tol)
    println("  xi  = ", xi)
    println("  |X|  = ", length(X))
    println("  |X0| = ", length(X0))
    println("  |Xu| = ", length(Xu))
    println("  nodes = ", nodes)
    println("  edges = ", edges)

    println("\nBasis order from JSON:")
    for (k, s) in enumerate(basis_strings)
        println("  ", k, ": ", s, " -> ", exponents[k])
    end

    println("\nValidating P1: one-step closure...")
    p1_min, p1_worst = validate_P1(coeff_values, exponents, edges, X)
    @printf("P1 minimum value = %.16e\n", p1_min)
    println("P1 status = ", pass_fail(p1_min, tol))
    print_worst("P1 worst case:", p1_worst)

    println("\nValidating P2: transitive implication on active premises...")
    p2_count, p2_min, p2_worst = validate_P2(coeff_values, exponents, edges, nodes, X, tol)
    println("P2 active premise count = ", p2_count)
    if p2_count == 0
        println("P2 minimum conclusion over active premises = none")
        println("P2 status = WARNING: no active premises on the grid")
    else
        @printf("P2 minimum conclusion over active premises = %.16e\n", p2_min)
        println("P2 status = ", pass_fail(p2_min, tol))
    end
    print_worst("P2 worst active case:", p2_worst)

    println("\nValidating P3: well-foundedness decrease...")
    p3_count,
    p3_exact_count,
    p3_min_uncond,
    p3_min_active,
    p3_min_exact_active,
    p3_worst_uncond,
    p3_worst_active,
    p3_worst_exact_active =
        validate_P3(coeff_values, exponents, nodes, X0, Xu, xi, tol)

    @printf("P3 minimum unconditional decrease slack = %.16e\n", p3_min_uncond)
    println("P3 unconditional status = ", pass_fail(p3_min_uncond, tol))

    println("P3 conservative active premise count = ", p3_count)
    if p3_count == 0
        println("P3 minimum conservative-active decrease slack = none")
        println("P3 conservative-active status = WARNING: no active premises on the grid")
    else
        @printf("P3 minimum conservative-active decrease slack = %.16e\n", p3_min_active)
        println("P3 conservative-active status = ", pass_fail(p3_min_active, tol))
    end

    println("P3 exact active premise count = ", p3_exact_count)
    if p3_exact_count == 0
        println("P3 minimum exact-active decrease slack = none")
        println("P3 exact-active status = WARNING: no exact-active premises on the grid")
    else
        @printf("P3 minimum exact-active decrease slack = %.16e\n", p3_min_exact_active)
        println("P3 exact-active status = ", pass_fail(p3_min_exact_active, tol))
    end

    print_worst("P3 worst unconditional case:", p3_worst_uncond)
    print_worst("P3 worst conservative-active case:", p3_worst_active)
    print_worst("P3 worst exact-active case:", p3_worst_exact_active)

    conservative_p3_pass = p3_count > 0 && p3_min_active >= -tol
    exact_p3_pass = p3_exact_count > 0 && p3_min_exact_active >= -tol

    all_pass = (p1_min >= -tol) &&
            (p2_count > 0 && p2_min >= -tol) &&
            exact_p3_pass

    preferred_pass = all_pass && conservative_p3_pass
    strong_p3 = p3_min_uncond >= -tol

    println("\nSummary")
    println("  Required validation with exact-active P3 = ", all_pass ? "PASS" : "FAIL")
    println("  Preferred validation with conservative-active P3 = ", preferred_pass ? "PASS" : "FAIL")
    println("  Strong unconditional P3 = ", strong_p3 ? "PASS" : "FAIL")

    out = Dict(
        "certificate_path" => cert_path,
        "N1" => N1,
        "N2" => N2,
        "tol" => tol,
        "xi" => xi,
        "num_X" => length(X),
        "num_X0" => length(X0),
        "num_Xu" => length(Xu),
        "P1_minimum" => p1_min,
        "P1_pass" => p1_min >= -tol,
        "P2_active_premise_count" => p2_count,
        "P2_minimum_active_conclusion" => p2_count == 0 ? nothing : p2_min,
        "P2_pass" => p2_count > 0 && p2_min >= -tol,
        "P3_minimum_unconditional_decrease_slack" => p3_min_uncond,
        "P3_unconditional_pass" => p3_min_uncond >= -tol,
        "P3_conservative_active_premise_count" => p3_count,
        "P3_minimum_conservative_active_decrease_slack" => p3_count == 0 ? nothing : p3_min_active,
        "P3_conservative_active_pass" => conservative_p3_pass,

        "P3_exact_active_premise_count" => p3_exact_count,
        "P3_minimum_exact_active_decrease_slack" => p3_exact_count == 0 ? nothing : p3_min_exact_active,
        "P3_exact_active_pass" => exact_p3_pass,

        "required_validation_exact_active_pass" => all_pass,
        "preferred_validation_conservative_active_pass" => preferred_pass,
        "strong_unconditional_P3_pass" => strong_p3,
    )

    mkpath(joinpath("results", "validation"))
    out_path = joinpath("results", "validation", "pccc_validate_platoon_sos_summary.json")
    open(out_path, "w") do io
        JSON.print(io, out, 2)
    end
    println("\nSaved validation summary: ", out_path)

    if !all_pass
        exit(1)
    end
end

main()
