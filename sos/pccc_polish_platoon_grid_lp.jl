# Finite-grid coefficient polishing for the platoon PC-CC candidate, v3.
#
# This script is diagnostic only: it repairs/assesses a quadratic PC-CC on a
# finite validation grid. A polished grid candidate is NOT an SOS/TSSOS proof.
#
# Differences from v2:
#   - never silently reports the unchanged seed as a polished result;
#   - writes a status JSON explaining whether any LP iterate was accepted;
#   - saves every LP candidate separately;
#   - rescans each LP candidate and selects by independent grid violation score;
#   - uses explicit JuMP.MOI import.
#
# Run:
#   julia --project=. sos/pccc_polish_platoon_grid_lp.jl \
#       results/pccc_synth_platoon_tssos_alternating_best.json 21 41
#
# Outputs:
#   results/pccc_synth_platoon_grid_polished_v3_roundXX.json
#   results/pccc_synth_platoon_grid_polished.json   only if an LP candidate is accepted
#   results/pccc_synth_platoon_grid_polishing_status.json

using JSON
using JuMP
using MosekTools
using Printf

const MOI = JuMP.MOI

const DEFAULT_INPUT = joinpath("results", "pccc_synth_platoon_tssos_alternating_best.json")
const DEFAULT_N1 = 21
const DEFAULT_N2 = 41

# Match validator settings.
const XI_DEFAULT = 1.0e-6
const ACTIVE_TOL = 1.0e-7

# Set margins to zero for first repair. If zero-margin repair works, margins can
# be increased later. The validator itself uses tol=1e-7.
const GRID_MARGIN_P1 = 0.0
const GRID_MARGIN_P2 = 0.0
const GRID_MARGIN_P3 = 0.0

const MAX_ROUNDS = 8
const MAX_NEW_P1_CUTS = 2000
const MAX_NEW_P2_CUTS = 3000
const MAX_NEW_P3_CUTS = 3000

# Trust region around seed. The best TSSOS candidate has coefficients around
# 1e-6 to 1e-5, but a repair of order 1e-4 may be needed. Start generous.
const INITIAL_RHO_UB = 5.0e-3
const C_ABS_BOUND = 1.0e2

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

function coeff_values_for_json(C)
    nbasis = size(C, 1)
    out = Vector{Any}(undef, nbasis)
    for k in 1:nbasis
        out[k] = [[C[k,p,q] for p in 1:2] for q in 1:2]
    end
    return out
end

function C_value(C, exponents, p::Int, q::Int, a1::Float64, a2::Float64, b1::Float64, b2::Float64)
    phi = basis_values(exponents, a1, a2, b1, b2)
    s = 0.0
    @inbounds for k in eachindex(phi)
        s += C[k,p,q] * phi[k]
    end
    return s
end

struct Cut
    ks::Vector{Int}
    ps::Vector{Int}
    qs::Vector{Int}
    alpha::Vector{Float64}
    rhs::Float64
    label::String
    score::Float64
end

function p1_cut(exponents, p, q, x, fx; margin=GRID_MARGIN_P1, score=0.0)
    phi = basis_values(exponents, x[1], x[2], fx[1], fx[2])
    nb = length(phi)
    return Cut(collect(1:nb), fill(p, nb), fill(q, nb), phi, margin,
               "P1 edge=($p,$q) x=$x", score)
end

function p2_cut(exponents, p, r, x, y; margin=GRID_MARGIN_P2, score=0.0)
    phi = basis_values(exponents, x[1], x[2], y[1], y[2])
    nb = length(phi)
    return Cut(collect(1:nb), fill(p, nb), fill(r, nb), phi, margin,
               "P2 target=($p,$r) x=$x y=$y", score)
end

function p3_cut(exponents, p, q, r, x0, y, z, xi; margin=GRID_MARGIN_P3, score=0.0)
    phi_a = basis_values(exponents, x0[1], x0[2], y[1], y[2])
    phi_c = basis_values(exponents, x0[1], x0[2], z[1], z[2])
    nb = length(phi_a)
    ks = Int[]; ps = Int[]; qs = Int[]; alpha = Float64[]
    for k in 1:nb
        push!(ks, k); push!(ps, p); push!(qs, q); push!(alpha, phi_a[k])
        push!(ks, k); push!(ps, p); push!(qs, r); push!(alpha, -phi_c[k])
    end
    return Cut(ks, ps, qs, alpha, xi + margin,
               "P3 pqr=($p,$q,$r) x0=$x0 y=$y z=$z", score)
end

function push_top!(vec::Vector{Cut}, cut::Cut, maxlen::Int)
    push!(vec, cut)
    if length(vec) > 3 * maxlen
        sort!(vec, by = c -> c.score)
        resize!(vec, maxlen)
    end
end

function finish_top!(vec::Vector{Cut}, maxlen::Int)
    sort!(vec, by = c -> c.score)
    if length(vec) > maxlen
        resize!(vec, maxlen)
    end
    return vec
end

function add_cut!(model, cvar, cut::Cut)
    expr = zero(AffExpr)
    @inbounds for i in eachindex(cut.alpha)
        expr += cut.alpha[i] * cvar[cut.ks[i], cut.ps[i], cut.qs[i]]
    end
    @constraint(model, expr >= cut.rhs)
end

function scan_violations(C, exponents, edges, nodes, X, X0, Xu, xi)
    p1 = Cut[]; p2 = Cut[]; p3 = Cut[]
    min_p1 = Inf; min_p2 = Inf; min_p3 = Inf

    for (p, sigma, q) in edges
        for x in X
            fx = f_sigma(sigma, x[1], x[2])
            val = C_value(C, exponents, p, q, x[1], x[2], fx[1], fx[2])
            min_p1 = min(min_p1, val)
            if val < GRID_MARGIN_P1
                push_top!(p1, p1_cut(exponents, p, q, x, fx; score=val-GRID_MARGIN_P1), MAX_NEW_P1_CUTS)
            end
        end
    end

    for (p, sigma, q) in edges
        for r in nodes, x in X
            fx = f_sigma(sigma, x[1], x[2])
            for y in X
                A = C_value(C, exponents, q, r, fx[1], fx[2], y[1], y[2])
                if A >= -ACTIVE_TOL
                    B = C_value(C, exponents, p, r, x[1], x[2], y[1], y[2])
                    min_p2 = min(min_p2, B)
                    if B < GRID_MARGIN_P2
                        push_top!(p2, p2_cut(exponents, p, r, x, y; score=B-GRID_MARGIN_P2), MAX_NEW_P2_CUTS)
                    end
                end
            end
        end
    end

    for p in nodes, q in nodes, r in nodes
        for x0 in X0, y in Xu
            A = C_value(C, exponents, p, q, x0[1], x0[2], y[1], y[2])
            if A >= -ACTIVE_TOL
                for z in Xu
                    B = C_value(C, exponents, q, r, y[1], y[2], z[1], z[2])
                    if B >= -ACTIVE_TOL
                        Cpr = C_value(C, exponents, p, r, x0[1], x0[2], z[1], z[2])
                        D = A - xi - Cpr
                        min_p3 = min(min_p3, D)
                        if D < GRID_MARGIN_P3
                            push_top!(p3, p3_cut(exponents, p, q, r, x0, y, z, xi; score=D-GRID_MARGIN_P3), MAX_NEW_P3_CUTS)
                        end
                    end
                end
            end
        end
    end

    p1 = finish_top!(p1, MAX_NEW_P1_CUTS)
    p2 = finish_top!(p2, MAX_NEW_P2_CUTS)
    p3 = finish_top!(p3, MAX_NEW_P3_CUTS)
    worst_violation = max(GRID_MARGIN_P1 - min_p1, GRID_MARGIN_P2 - min_p2, GRID_MARGIN_P3 - min_p3)
    metrics = Dict(
        "P1_minimum" => min_p1,
        "P2_minimum_active_conclusion" => min_p2,
        "P3_minimum_exact_active_decrease_slack" => min_p3,
        "worst_violation" => worst_violation,
        "num_P1_cuts" => length(p1),
        "num_P2_cuts" => length(p2),
        "num_P3_cuts" => length(p3),
    )
    return p1, p2, p3, metrics
end

function solve_polish_lp(Cseed, cuts::Vector{Cut}; rho_ub=INITIAL_RHO_UB)
    nbasis = size(Cseed, 1)
    model = Model(Mosek.Optimizer)
    set_silent(model)

    @variable(model, c[1:nbasis, 1:2, 1:2])
    @variable(model, 0 <= rho <= rho_ub)

    for k in 1:nbasis, p in 1:2, q in 1:2
        @constraint(model, c[k,p,q] - Cseed[k,p,q] <= rho)
        @constraint(model, Cseed[k,p,q] - c[k,p,q] <= rho)
        @constraint(model, c[k,p,q] <= C_ABS_BOUND)
        @constraint(model, c[k,p,q] >= -C_ABS_BOUND)
    end

    for cut in cuts
        add_cut!(model, c, cut)
    end

    @objective(model, Min, rho)
    optimize!(model)

    term = termination_status(model)
    prim = primal_status(model)
    println("LP termination = ", term, ", primal = ", prim)
    if !(prim in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT))
        return nothing, string(term), string(prim), nothing
    end

    Cout = zeros(Float64, nbasis, 2, 2)
    for k in 1:nbasis, p in 1:2, q in 1:2
        Cout[k,p,q] = value(c[k,p,q])
    end
    return Cout, string(term), string(prim), value(rho)
end

function save_candidate(template_data, C, out_path, round, rho, metrics)
    data = deepcopy(template_data)
    data["coeff_values"] = coeff_values_for_json(C)
    data["candidate_saved"] = true
    data["feasible_certificate_saved"] = false
    data["paper_ready_solver_status"] = false
    data["certificate_type"] = "grid_polished_quadratic_candidate"
    data["certificate_warning"] = "Finite-grid polished candidate only. This is not an SOS/TSSOS proof. Re-certify before using as a proof."
    data["grid_polishing"] = Dict(
        "round" => round,
        "rho" => rho,
        "N1" => DEFAULT_N1,
        "N2" => DEFAULT_N2,
        "P1_margin" => GRID_MARGIN_P1,
        "P2_margin" => GRID_MARGIN_P2,
        "P3_margin" => GRID_MARGIN_P3,
        "active_tolerance" => ACTIVE_TOL,
        "metrics" => metrics,
    )
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, data, 2)
    end
    println("Saved candidate: ", out_path)
end

function json_safe(x)
    if x isa AbstractFloat
        if isnan(x) || isinf(x)
            return nothing
        else
            return x
        end
    elseif x isa Dict
        return Dict(k => json_safe(v) for (k,v) in x)
    elseif x isa AbstractVector
        return [json_safe(v) for v in x]
    else
        return x
    end
end

function save_status(out_path, status)
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, json_safe(status), 2)
    end
    println("Saved polishing status: ", out_path)
end

function main()
    input_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_INPUT
    N1 = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : DEFAULT_N1
    N2 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : DEFAULT_N2
    if !isfile(input_path)
        error("Input candidate not found: $input_path")
    end

    println("Loading seed candidate: ", input_path)
    data = JSON.parsefile(input_path)
    basis_strings = data["basis_xy"]
    exponents = [monomial_exponents(s) for s in basis_strings]
    Cseed = load_coeff_array(data)
    Ccur = copy(Cseed)
    nodes = [Int(v) for v in data["nodes"]]
    edges = [(Int(e[1]), Int(e[2]), Int(e[3])) for e in data["edges"]]
    xi = haskey(data, "wf_decrease") ? Float64(data["wf_decrease"]) : XI_DEFAULT

    X = make_X_grid(N1, N2)
    X0 = make_X_grid(N1, N2; gap_max=0.3)
    Xu = make_X_grid(N1, N2; gap_max=0.4)
    println("Grid sizes: |X|=", length(X), ", |X0|=", length(X0), ", |Xu|=", length(Xu))

    cuts = Cut[]
    seen = Set{String}()
    accepted_any_lp = false
    best_C = nothing
    best_metrics = nothing
    best_round = 0
    best_rho = nothing
    history = Any[]

    # Seed metrics, for comparison only.
    p1, p2, p3, seed_metrics = scan_violations(Ccur, exponents, edges, nodes, X, X0, Xu, xi)
    println("Seed metrics: ", seed_metrics)
    push!(history, Dict("round" => 0, "type" => "seed", "metrics" => seed_metrics))

    for round in 1:MAX_ROUNDS
        println("\n==============================")
        println("Polishing round ", round)
        p1, p2, p3, metrics = scan_violations(Ccur, exponents, edges, nodes, X, X0, Xu, xi)
        @printf("Current minima: P1 %.6e, P2 %.6e, P3 %.6e, worst violation %.6e\n",
                metrics["P1_minimum"], metrics["P2_minimum_active_conclusion"],
                metrics["P3_minimum_exact_active_decrease_slack"], metrics["worst_violation"])
        if metrics["worst_violation"] <= 0.0
            println("No finite-grid violations remain.")
            break
        end

        new_count = 0
        for cut in vcat(p1, p2, p3)
            if !(cut.label in seen)
                push!(cuts, cut)
                push!(seen, cut.label)
                new_count += 1
            end
        end
        println("Total cuts = ", length(cuts), " (new this round = ", new_count, ")")
        if new_count == 0
            println("No new cuts; stopping.")
            break
        end

        Cnext, term, prim, rho = solve_polish_lp(Cseed, cuts)
        push!(history, Dict("round" => round, "term" => term, "primal" => prim, "rho" => rho, "num_cuts" => length(cuts)))
        if Cnext === nothing
            println("LP did not return a feasible point. Stopping without overwriting final polished JSON.")
            break
        end
        @printf("LP rho = %.6e\n", rho)

        # Evaluate and save this LP candidate.
        _, _, _, cand_metrics = scan_violations(Cnext, exponents, edges, nodes, X, X0, Xu, xi)
        @printf("Candidate minima: P1 %.6e, P2 %.6e, P3 %.6e, worst violation %.6e\n",
                cand_metrics["P1_minimum"], cand_metrics["P2_minimum_active_conclusion"],
                cand_metrics["P3_minimum_exact_active_decrease_slack"], cand_metrics["worst_violation"])

        out_iter = joinpath("results", @sprintf("pccc_synth_platoon_grid_polished_v3_round%02d.json", round))
        save_candidate(data, Cnext, out_iter, round, rho, cand_metrics)
        accepted_any_lp = true

        if best_metrics === nothing || cand_metrics["worst_violation"] < best_metrics["worst_violation"]
            best_C = copy(Cnext)
            best_metrics = deepcopy(cand_metrics)
            best_round = round
            best_rho = rho
        end

        Ccur = Cnext
        if cand_metrics["worst_violation"] <= 0.0
            println("Accepted LP candidate satisfies all selected finite-grid checks.")
            break
        end
    end

    status = Dict(
        "input_path" => input_path,
        "N1" => N1,
        "N2" => N2,
        "accepted_any_lp" => accepted_any_lp,
        "seed_metrics" => seed_metrics,
        "best_round" => best_round,
        "best_rho" => best_rho,
        "best_metrics" => best_metrics,
        "history" => history,
    )
    save_status(joinpath("results", "pccc_synth_platoon_grid_polishing_status.json"), status)

    if accepted_any_lp && best_C !== nothing
        out_path = joinpath("results", "pccc_synth_platoon_grid_polished.json")
        save_candidate(data, best_C, out_path, best_round, best_rho, best_metrics)
        println("\nSelected best LP candidate: ", out_path)
        println("Now run the standard validator on: ", out_path)
    else
        println("\nNo LP candidate was accepted. The previous pccc_synth_platoon_grid_polished.json, if present, was not overwritten.")
    end
end

main()
