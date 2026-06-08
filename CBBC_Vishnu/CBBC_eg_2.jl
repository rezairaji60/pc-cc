using JuMP
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using LinearAlgebra
using TSSOS # important for SOS, see https://github.com/wangjie212/TSSOS
using Latexify
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

aut_states = [0,1,2]
X_c1 = [3,4]
X_c2 =  [-1,1]

X_b1 = [2,4]
X_b2 = [-2.5,-2]


#initial, unsafe sets and state space
Xo1 = [3,3.5] # initial set
Xo2 = [1.5,2]
X1 = [-2.5,4]
X2 = [-2.5,2.5]
# XVFx = [3,4]
# XVFy = [-1,1]


# semi-algebraic set description polynomials, g_i(x)>=0

g_init = [x-Xo1[1], Xo1[2]-x, y - Xo2[1], Xo2[2] - y] 
g_state_set = [x-X1[1], X1[2]-x, y - X2[1], X2[2] - y]

g_c = [x - X_c1[1], X_c1[2] - x, y - X_c2[1], X_c2[2] - y]
g_a = [x - X_b1[1], X_b1[2] - x, y - X_b2[1], X_b2[2] - y]

g_b = [[-g_i] for g_i in g_c for g_i in g_a]

function CBBC_comp(deg,k)
    model = Model(optimizer_with_attributes(Mosek.Optimizer))
    set_optimizer_attribute(model, MOI.Silent(), true)
B_templates =[]
B_coeff_list =[]
B_f_list = []
B_monom_list =[]
for i in 0:2
    B_counter_templates = []
    B_counter_coeff_list =[]
    B_counter_f_list = []
    B_counter_monom_list = []
    for j in 0:k
        B, Bc, Bb = add_poly!(model,vars, deg)
        Bf = B(vars=>f) #B(f(x)) https://juliapackages.com/p/dynamicpolynomials
        push!(B_counter_templates,B)
        push!(B_counter_coeff_list,Bc)
        push!(B_counter_f_list, Bf)
        push!(B_counter_monom_list, Bb)
    end
    push!(B_templates,B_counter_templates)
    push!(B_coeff_list, B_counter_coeff_list)
    push!(B_f_list, B_counter_f_list)
    push!(B_monom_list, B_counter_monom_list)
end

for i in 0:2
    _ = add_psatz!(model, -B_templates[i+1][1], vars, g_init, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)    
end
for i in 1:2
       _ = add_psatz!(model, B_templates[i+1][k+1] - 2*ϵ, vars, g_state_set, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
end

    for i = 1:(k)
        _ = add_psatz!(model,B_templates[1][i] - B_f_list[2][i], vars, g_a, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
        _ = add_psatz!(model,B_templates[2][i] - B_f_list[2][i+1], vars, g_a, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
        _ = add_psatz!(model,B_templates[3][i] - B_f_list[3][i+1], vars, g_a, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)

        _ = add_psatz!(model,B_templates[1][i] - B_f_list[2][i], vars, g_c, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
        _ = add_psatz!(model,B_templates[2][i] - B_f_list[2][i+1], vars, g_c, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
        _ = add_psatz!(model,B_templates[3][i] - B_f_list[2][i+1], vars, g_c, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)

        for t in g_b
                    _ = add_psatz!(model,B_templates[2][i] - B_f_list[1][i+1], vars, t, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
                    _ = add_psatz!(model,B_templates[1][i] - B_f_list[1][i], vars, t, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
                    _ = add_psatz!(model,B_templates[3][i] - B_f_list[3][i+1], vars, t, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
        end
    end
    _ = add_psatz!(model,B_templates[1][k+1] - B_f_list[2][k+1], vars, g_a, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
    _ = add_psatz!(model,B_templates[1][k+1] - B_f_list[2][k+1], vars, g_c, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
    for t in g_b
        _ = add_psatz!(model,B_templates[1][k+1] - B_f_list[1][k+1], vars, t, [], div(deg+sos_tol,2), QUIET=true, CS=false, TS=false, GroebnerBasis=true)
    end
    optimize!(model) #solve for coefficients
    status = termination_status(model)  #all_variables(model)
#println(MosekTools.getprimalobj(model), MosekTools.getdualobj(model))
Barrier_list = []
    for i in aut_states
        B_list_aut_state = []
        for tk in 1:(k+1)
            Barrier_coeffs = value.(B_coeff_list[i+1][tk]) # get the values of each coefficient for B_i
        for j = eachindex(Barrier_coeffs)
            Barrier_coeffs[j] = round(Barrier_coeffs[j]; digits = error) # round to order of error
        end
        push!(B_list_aut_state,Barrier_coeffs'*B_monom_list[i+1][tk])
    end
    push!(Barrier_list, B_list_aut_state)
end
    return(status,Barrier_list)
end

max_deg = 6
k = 2
for tk = 1:k
for deg = 1:max_deg
    file = open("./CBBC/CBBC_2_deg="*string(deg)*"_k="*string(tk)*".txt", "w");
    stats = @timed data = CBBC_comp(deg, tk)
    status, BC_list = data
    write(file, "poly deg := "*string(deg)*"\t"*"\n")
    write(file, "status: "*string(status)*"\n")
    for i in aut_states
        write(file,"aut_state = "*string(i)*"\n")
        for j = 1:tk+1
            write(file, "k ="*string(j)*"\n")
            write(file, latexify(string(BC_list[i+1][j]))*"\n")
        end
    end
    write(file, "time: "*string(stats.time)*"\n\n")
    close(file)
    end
end
println("Finished")
