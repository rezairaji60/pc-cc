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


# vector field
@polyvar x y 
vars = [x, y]

# simple dynamics. same as the z3_smt_barrier example
f = [
    x + 0.1*y,
    y + 0.1*(-x + 0.4*y*(1 - x*x))
]


#initial, unsafe sets and state space
Xo1 = [3,3.5] # initial set
Xo2 = [1.5,2]
X1 = [-2.5,4]
X2 = [-2.5,2.5]
XVFx = [3,4]
XVFy = [-1,0]


# semi-algebraic set description polynomial of the initial/unsafe set and state space, >=0 by default
g_init = [x-Xo1[1], Xo1[2]-x, y - Xo2[1], Xo2[2] - y] #initial
g_state_set = [x-X1[1], X1[2]-x, y - X2[1], X2[2] - y] #whole

finite_visit_polys = [x - XVFx[1], XVFx[2] - x, y - XVFy[1], y - XVFy[2]]
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

    model, _ = add_psatz!(model, -B_templates[1], vars, g_init, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    #model,info11 = add_psatz!(model,  B - eps_1, vars, g_unsafe_1, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    # 2nd condition for B_0, unsafe states
    model, _ = add_psatz!(model, B_templates[k+1] - 2*ϵ, vars, g_state_set, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    for i = 1:k
        model, _ = add_psatz!(model,B_templates[i] - B_f_list[i+1], vars, g_finite_visit, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
    end
    for i = 1:k+1
        for t in g_finit_visit_comp
                    model, _ = add_psatz!(model,B_templates[i] - B_f_list[i], vars, t, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, Groebnerbasis=true)
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
k = 8
for tk = 1:k
    file = open("./systems/CBBC/CBBC_1"*string(tk)*".txt", "w");
    for deg = 1:max_deg
        stats = @timed data = CBBC_comp(deg, tk)
        status, BC_list = data
        for j=1:tk
        write(file, "poly deg: "*string(deg)*"\t"*string(j)*"\n")
        write(file, "status: "*string(status)*"\n")
        write(file, Base.replace(string(BC_list[j]),"e"=>"*10^")*"\n")
        write(file, "time: "*string(stats.time)*"\n\n")
        end
    end
    close(file)
end
println("Finished")
