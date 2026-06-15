# Gap-sweep validator for the platoon PC-CC candidate.
#
# Diagnostic purpose:
#   Reuse an existing quadratic PC-CC JSON candidate and evaluate dense-grid
#   P1/P2/P3 metrics while varying only the persistence-region gap bound
#   Xu = {0 <= x1 <= 3, 0 <= x2 <= 5.1, 0 <= x2-x1 <= gap_max}.
#
# Run:
#   julia --project=. sos/pccc_validate_platoon_gap_sweep.jl \
#       results/pccc_synth_platoon_grid_polished_best.json 21 41
#
# Outputs:
#   results/validation/pccc_validate_platoon_gap_sweep.json
#   results/validation/pccc_validate_platoon_gap_sweep.csv

using JSON
using Printf

const ACTIVE_TOL = 1.0e-7
const XI_DEFAULT = 1.0e-6
const GAP_LIST_DEFAULT = [0.40, 0.375, 0.35, 0.325, 0.30, 0.275, 0.25]

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
                if gap_max === nothing || gap <= Float64(gap_max) + 1.0e-12
                    push!(pts, (Float64(x1), Float64(x2)))
                end
            end
        end
    end
    return pts
end

function normalize_monomial_string(s::AbstractString)
    t = replace(String(s), " " => "")
    t = replace(t, "·" => "*")
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

function load_coeff_array(data)
    coeff_values = data["coeff_values"]
    nbasis = length(data["basis_xy"])
    C = zeros(Float64, nbasis, 2, 2)
    if length(coeff_values) == nbasis
        for k in 1:nbasis, q in 1:2, p in 1:2
            C[k,p,q] = Float64(coeff_values[k][q][p])
        end
    else
        for p in 1:2, q in 1:2, k in 1:nbasis
            C[k,p,q] = Float64(coeff_values[p][q][k])
        end
    end
    return C
end

function C_value(C, exponents, p::Int, q::Int, a1::Float64, a2::Float64, b1::Float64, b2::Float64)
    phi = basis_values(exponents, a1, a2, b1, b2)
    s = 0.0
    @inbounds for k in eachindex(phi)
        s += C[k,p,q] * phi[k]
    end
    return s
end

function scan_p1(C, exponents, edges, X)
    min_p1 = Inf
    worst = nothing
    for (p, sigma, q) in edges
        for x in X
            fx = f_sigma(sigma, x[1], x[2])
            val = C_value(C, exponents, p, q, x[1], x[2], fx[1], fx[2])
            if val < min_p1
                min_p1 = val
                worst = Dict("edge" => [p, sigma, q], "x" => collect(x), "fx" => [fx[1], fx[2]], "value" => val)
            end
        end
    end
    return min_p1, worst
end

function scan_p2(C, exponents, edges, nodes, X)
    min_p2 = Inf
    active = 0
    worst = nothing
    for (p, sigma, q) in edges
        for r in nodes, x in X
            fx = f_sigma(sigma, x[1], x[2])
            for y in X
                A = C_value(C, exponents, q, r, fx[1], fx[2], y[1], y[2])
                if A >= -ACTIVE_TOL
                    active += 1
                    B = C_value(C, exponents, p, r, x[1], x[2], y[1], y[2])
                    if B < min_p2
                        min_p2 = B
                        worst = Dict("edge" => [p, sigma, q], "r" => r, "x" => collect(x), "y" => collect(y), "A" => A, "B" => B)
                    end
                end
            end
        end
    end
    return min_p2, active, worst
end

function scan_p3(C, exponents, nodes, X0, Xu, xi)
    min_uncond = Inf
    min_active = Inf
    active = 0
    worst_uncond = nothing
    worst_active = nothing
    for p in nodes, q in nodes, r in nodes
        for x0 in X0, y in Xu, z in Xu
            A = C_value(C, exponents, p, q, x0[1], x0[2], y[1], y[2])
            B = C_value(C, exponents, q, r, y[1], y[2], z[1], z[2])
            Cpr = C_value(C, exponents, p, r, x0[1], x0[2], z[1], z[2])
            D = A - xi - Cpr
            if D < min_uncond
                min_uncond = D
                worst_uncond = Dict("p" => p, "q" => q, "r" => r, "x0" => collect(x0), "y" => collect(y), "z" => collect(z), "A" => A, "B" => B, "D" => D)
            end
            if A >= -ACTIVE_TOL && B >= -ACTIVE_TOL
                active += 1
                if D < min_active
                    min_active = D
                    worst_active = Dict("p" => p, "q" => q, "r" => r, "x0" => collect(x0), "y" => collect(y), "z" => collect(z), "A" => A, "B" => B, "D" => D)
                end
            end
        end
    end
    return min_uncond, min_active, active, worst_uncond, worst_active
end

function bool_pass(v; tol=ACTIVE_TOL)
    return v >= -tol
end

function main()
    input_path = length(ARGS) >= 1 ? ARGS[1] : joinpath("results", "pccc_synth_platoon_grid_polished_best.json")
    N1 = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 21
    N2 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 41
    xi = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : XI_DEFAULT
    gaps = length(ARGS) >= 5 ? [parse(Float64, s) for s in split(ARGS[5], ",")] : GAP_LIST_DEFAULT

    println("Loading certificate: ", input_path)
    data = JSON.parsefile(input_path)
    basis = String.(data["basis_xy"])
    exponents = [monomial_exponents(b) for b in basis]
    C = load_coeff_array(data)

    nodes = [1, 2]
    edges = [(1,1,1), (1,1,2), (1,2,2), (2,2,1)]
    X = make_X_grid(N1, N2)
    X0 = make_X_grid(N1, N2; gap_max=0.3)

    println("Dense-grid gap sweep")
    println("  N1 = $N1, N2 = $N2, xi = $xi, active_tol = $ACTIVE_TOL")
    println("  |X| = $(length(X)), |X0| = $(length(X0))")

    # P1/P2 are independent of Xu gap; compute once.
    @printf("\nComputing P1/P2 once on X...\n")
    p1_min, p1_worst = scan_p1(C, exponents, edges, X)
    p2_min, p2_active, p2_worst = scan_p2(C, exponents, edges, nodes, X)

    rows = Vector{Dict{String, Any}}()
    @printf("\n%-9s %7s %12s %12s %12s %12s %10s %10s\n", "gap_max", "|Xu|", "P1_min", "P2_min", "P3_exact", "P3_uncond", "P3_active", "pass")
    @printf("%s\n", repeat("-", 94))

    for gap in gaps
        Xu = make_X_grid(N1, N2; gap_max=gap)
        p3_uncond, p3_exact, p3_active, worst_uncond, worst_exact = scan_p3(C, exponents, nodes, X0, Xu, xi)
        req_pass = bool_pass(p1_min) && bool_pass(p2_min) && bool_pass(p3_exact)
        @printf("%-9.3f %7d %12.4e %12.4e %12.4e %12.4e %10d %10s\n",
                gap, length(Xu), p1_min, p2_min, p3_exact, p3_uncond, p3_active, req_pass ? "PASS" : "FAIL")
        push!(rows, Dict(
            "gap_max" => gap,
            "num_Xu" => length(Xu),
            "P1_minimum" => p1_min,
            "P1_pass" => bool_pass(p1_min),
            "P2_minimum_active_conclusion" => p2_min,
            "P2_active_premise_count" => p2_active,
            "P2_pass" => bool_pass(p2_min),
            "P3_minimum_unconditional_decrease_slack" => p3_uncond,
            "P3_minimum_exact_active_decrease_slack" => p3_exact,
            "P3_exact_active_premise_count" => p3_active,
            "P3_unconditional_pass" => bool_pass(p3_uncond),
            "P3_exact_active_pass" => bool_pass(p3_exact),
            "required_validation_exact_active_pass" => req_pass,
            "worst_P3_unconditional_case" => worst_uncond,
            "worst_P3_exact_active_case" => worst_exact,
        ))
    end

    outdir = joinpath("results", "validation")
    mkpath(outdir)
    json_path = joinpath(outdir, "pccc_validate_platoon_gap_sweep.json")
    csv_path = joinpath(outdir, "pccc_validate_platoon_gap_sweep.csv")
    status = Dict(
        "certificate_path" => input_path,
        "N1" => N1,
        "N2" => N2,
        "xi" => xi,
        "active_tol" => ACTIVE_TOL,
        "num_X" => length(X),
        "num_X0" => length(X0),
        "P1_worst_case" => p1_worst,
        "P2_worst_case" => p2_worst,
        "rows" => rows,
    )
    open(json_path, "w") do io
        JSON.print(io, status, 2)
    end
    open(csv_path, "w") do io
        println(io, "gap_max,num_Xu,P1_minimum,P2_minimum_active_conclusion,P3_minimum_exact_active_decrease_slack,P3_minimum_unconditional_decrease_slack,P3_exact_active_premise_count,required_validation_exact_active_pass")
        for row in rows
            println(io, join([
                row["gap_max"], row["num_Xu"], row["P1_minimum"], row["P2_minimum_active_conclusion"],
                row["P3_minimum_exact_active_decrease_slack"], row["P3_minimum_unconditional_decrease_slack"],
                row["P3_exact_active_premise_count"], row["required_validation_exact_active_pass"]
            ], ","))
        end
    end
    println("\nSaved gap sweep JSON: ", json_path)
    println("Saved gap sweep CSV:  ", csv_path)
end

main()
