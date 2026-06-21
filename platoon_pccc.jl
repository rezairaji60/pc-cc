using JuMP
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using LinearAlgebra
using TSSOS

println("Starting Nonlinear Platoon PC-CC SOS Simulation...")

# Configuration parameters
sos_tol = 1
tau_s2 = 1.0
tau_3a = 1.0
tau_3b = 1.0
error_digits = 4

# Polynomial variables
@polyvar x[1:2] y[1:2] yp[1:2] x0[1:2]

# Dynamics for Modes 1 and 2
function f1(v)
    return [0.01*v[2] + 0.9*v[1] - 0.02*v[1]^2,
            2.0 + 0.8*v[2] - 0.04*v[2]^2]
end

function f2(v)
    return [0.9*v[1] - 0.02*v[1]^2,
            2.0 + 0.8*v[2] - 0.04*v[2]^2]
end

f_sigma = [f1, f2]

# Domain inequalities (>= 0)
h_X(v)   = [v[1], 3 - v[1], v[2], 5.1 - v[2], v[2] - v[1]]
h_X0(v)  = [0.3 - (v[2] - v[1])]
h_Xvf(v) = [0.4 - (v[2] - v[1])]

# Graph definition
nodes = [1, 2]
# Edges: (p, sigma, q)
edges = [(1, 1, 1), (1, 2, 2), (2, 2, 1), (1, 1, 2)]

file = open("platoon_pccc_results.txt", "w")

for deg in 1:3
    println("=========================================")
    println("Testing Polynomial Certificate Degree: ", deg)
    println("=========================================")
    
    t_start = time()
    
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    # Uncomment to hide Mosek output
    # set_optimizer_attribute(model, MOI.Silent(), true)
    
    # Dictionary for certificates C_{p,q}
    C = Dict()
    C_vars = [x; y]
    
    for p in nodes, q in nodes
        poly, coef, mono = add_poly!(model, C_vars, deg)
        C[(p,q)] = (poly, coef, mono)
    end

    # SOS-1: One-step closure
    for (p, sigma, q) in edges
        poly = C[(p,q)][1]
        f_val = f_sigma[sigma](x)
        expr_sos1 = subs(poly, y => f_val)
        d1 = div(maxdegree(expr_sos1) + sos_tol, 2)
        add_psatz!(model, expr_sos1, x, h_X(x), [], d1, QUIET=true, CS=false, TS=false, GroebnerBasis=true)
    end

    # SOS-2: Transitive closure
    for (p, sigma, q) in edges
        for r in nodes
            poly_pr = C[(p,r)][1]
            poly_qr = C[(q,r)][1]
            f_val = f_sigma[sigma](x)
            expr_sos2 = poly_pr - tau_s2 * subs(poly_qr, x => f_val)
            d2 = div(maxdegree(expr_sos2) + sos_tol, 2)
            add_psatz!(model, expr_sos2, [x; y], [h_X(x); h_X(y)], [], d2, QUIET=true, CS=false, TS=false, GroebnerBasis=true)
        end
    end

    # SOS-3: Well-foundedness
    for p in nodes, q in nodes, r in nodes
        poly_pq = C[(p,q)][1]
        poly_pr = C[(p,r)][1]
        poly_qr = C[(q,r)][1]
        
        C_pq = subs(poly_pq, x => x0)
        C_pr = subs(poly_pr, x => x0, y => yp)
        C_qr = subs(poly_qr, x => y, y => yp)
        
        expr_sos3 = C_pq - C_pr - 1.0 - tau_3a * C_pq - tau_3b * C_qr
        
        vars_sos3 = [x0; y; yp]
        ineqs_sos3 = [h_X(x0); h_X0(x0); h_X(y); h_Xvf(y); h_X(yp); h_Xvf(yp)]
        
        d3 = div(maxdegree(expr_sos3) + sos_tol, 2)
        add_psatz!(model, expr_sos3, vars_sos3, ineqs_sos3, [], d3, QUIET=true, CS=false, TS=false, GroebnerBasis=true)
    end

    println("Optimizing...")
    optimize!(model)
    status = termination_status(model)
    t_elapsed = time() - t_start
    println("Solver Status: ", status)
    
    write(file, "poly deg: "*string(deg)*"\n")
    write(file, "status: "*string(status)*"\n")
    
    if status == MOI.OPTIMAL || status == MOI.ALMOST_OPTIMAL || status == MOI.SLOW_PROGRESS
        println("=> Found valid certificate for degree \$deg!")
        for p in nodes, q in nodes
            poly, coef, mono = C[(p,q)]
            val_coef = value.(coef)
            for i in eachindex(val_coef)
                val_coef[i] = round(val_coef[i]; digits = error_digits)
            end
            rounded_poly = sum(c * m for (c, m) in zip(val_coef, mono) if abs(c) > 0.0)
            println("C_{$p,$q}(x, y) = ", rounded_poly)
            
            write(file, "C_{$p,$q}(x,y): "*string(rounded_poly)*"\n")
            write(file, "C_{$p,$q} Monomials: "*string(mono)*"\n")
            write(file, "C_{$p,$q} Coefficients: "*string(val_coef)*"\n")
        end
    else
        println("=> Could not find certificate for degree \$deg.")
    end
    write(file, "time: "*string(t_elapsed)*"\n\n")
    println()
end

close(file)
