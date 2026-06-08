using JuMP
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using LinearAlgebra
using TSSOS # important for SOS, see https://github.com/wangjie212/TSSOS
ϵ = 10^(-3) # B(x) ≥ ϵ instead of B(x) > 0
eps_1 = 10^(-3) 
eps_2 = 10^(-3)
sos_tol = 8 # the maximum degree of unknown SOS polynomials = deg + sos_tol 
error = 6   # precision digit places
using Latexify


# vector field
@polyvar x y z
vars = [x, y, z]

T = 0.1
K = 0.06
c = -0.532
function mysin(x)
    return(x - x*x*x/6)
end
# simple dynamics. same as the z3_smt_barrier example
f = [
    x + T*K*(mysin(y - x) ) + c*x*x + 1,
    y + T*K*(mysin(x - y) + mysin(z-y)) + c*y*y + 1,
    z + T*K*mysin(y-z) + c*z*z + 1
]


#initial, unsafe sets and state space
X0 = [0,2*pi/15]
X = [0,2]

XVF = [0,0.7]

# semi-algebraic set description polynomial of the initial/unsafe set and state space, >=0 by default
g_init_1 = [i - X0[1] for i in vars] #initial
g_init_2 = [X0[2] - i for i in vars]
g_init = vcat(g_init_1, g_init_2)
g_state_set_1 = [i - X[1] for i in vars]
g_state_set_2 = [X[2] - i for i in vars]

g_state_set = vcat(g_state_set_1, g_state_set_2)

finite_visit_polys = [x, y, 0.7 - x, 0.7 - y, z, 2 - z]
g_finite_visit = [g_i - eps_2 for g_i in finite_visit_polys]
g_finit_visit_comp = [[ - g_i ] for g_i in finite_visit_polys ]
g_init_fv = [g_i for g_i in g_init for g_i in finite_visit_polys]

function CBBC_comp(deg,k)
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_optimizer_attribute(model, MOI.Silent(), true)
    B, Bc, Bb = add_poly!(model,vars, deg) # generate polynomial template with given variables and degree
    Bf = B(vars=>f) #B(f(x)) https://juliapackages.com/p/dynamicpolynomials
    B_templates  = [B] # store all unknown polynomial B(x) templates
    B_coeff_list = [Bc] # store all corresponding unknown polynomial coefficients
    B_f_list = [Bf] # store all corresponding unknown B(f(x)) templates
    B_monom_list = [Bb]
    for i = 2:(k+1)
        B, Bc, Bb = add_poly!(model,vars, deg) 
        Bf = B(vars=>f)
        push!(B_templates, B) # append to Bs in python
        push!(B_coeff_list, Bc)
        push!(B_f_list, Bf)
        push!(B_monom_list, Bb)
    end

    _ = add_psatz!(model, -B_templates[1], vars, g_init, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
    #model,info11 = add_psatz!(model,  B - eps_1, vars, g_unsafe_1, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
    # 2nd condition for B_0, unsafe states
    _ = add_psatz!(model, B_templates[k+1] - 2*ϵ, vars, g_state_set, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
    for i = 1:k
        _ = add_psatz!(model,B_templates[i] - B_f_list[i+1], vars, g_finite_visit, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
    end
    for i = 1:k+1
        for t in g_finit_visit_comp
                    _ = add_psatz!(model,B_templates[i] - B_f_list[i], vars, t, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
        end
    end
    optimize!(model) #solve for coefficients
    status = termination_status(model)  #all_variables(model)
#println(MosekTools.getprimalobj(model), MosekTools.getdualobj(model))
Barrier_list = []
    for i in eachindex(B_coeff_list)
        Barrier_coeffs = value.(B_coeff_list[i]) # get the values of each coefficient for B_i
        for j = eachindex(Barrier_coeffs)
            Barrier_coeffs[j] = round(Barrier_coeffs[j]; digits = error) # round to order of error
        end
        push!(Barrier_list,Barrier_coeffs'*B_monom_list[i])
    end
    return(status,Barrier_list)
end

max_deg = 6
k = 2
for tk = 1:k
    file = open("./CBBC/CBBC_3"*string(tk)*".txt", "w");
    for deg = 1:max_deg
        stats = @timed data = CBBC_comp(deg, tk)
        status, BC_list = data
        for j=1:tk
        write(file, "poly deg: "*string(deg)*"\t"*string(j)*"\n")
        write(file, "status: "*string(status)*"\n")
        write(file, Base.replace(latexify(string(BC_list[j])))*"\n")
        write(file, "time: "*string(stats.time)*"\n\n")
        end
    end
    close(file)
end
println("Finished")
