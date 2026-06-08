using JuMP
using MosekTools
using DynamicPolynomials
using MultivariatePolynomials
using SpecialFunctions
using TSSOS
using Plots

# Moments of the Lebesgue measure on the unit ball
function momball(a)
  n,m = size(a)
  y = zeros(m)
  for k = 1:m
    if all(.!Bool.(rem.(a[:,k],2)))
      y[k]=prod(gamma.((a[:,k].+1)/2))/gamma(1+(n+sum(a[:,k]))/2)
    end
  end
  return y
end

# Half-degree of polynomials w(x) and v(x)
d = 5

n = 2
@polyvar x[1:n] 
f = [x[1] + x[1]*x[2]; x[2] - x[1]^3] * 0.5
gS = 1 - x[1]^2 - x[2]^2
gB = gS
model = Model(optimizer_with_attributes(Mosek.Optimizer))
#set_optimizer_attribute(model, MOI.Silent(), true)

v, vc, vb = add_poly!(model, x, 2d)
w, wc, wb = add_poly!(model, x, 2d)
vf = subs(v, x[1]=>f[1], x[2]=>f[2])
dv = Int(ceil(maxdegree(vf)/2))

# Constraints
info1 = add_psatz!(model, vf, x, [gS], [], dv, QUIET=false, CS=false, TS=false, GroebnerBasis=true) # v o f >= 0 on S
info2 = add_psatz!(model, w-1-v, x, [gB], [], d, QUIET=false, CS=false, TS=false, GroebnerBasis=true) # w >= v + 1 on B
info3 = add_psatz!(model, w, x, [gB], [], d, QUIET=false, CS=false, TS=false, GroebnerBasis=true) # w >= 0 on B

supp = TSSOS.get_basis(n, 2d)

# Lebesgue moments on B

moment = momball(supp)
@objective(model, Min, moment'*wc) # minimization of int w d_lambda


optimize!(model)
status = termination_status(model)
if status != MOI.OPTIMAL
    println("termination status: $status")
    status = primal_status(model)
    println("solution status: $status")
end
objv = objective_value(model)
wp = value.(wc)'*wb

# Plot the superlevel set of w-1
x1 = range(-1, 1, length=1000)
x2 = range(-1, 1, length=1000)
hw(x1, x2) = if x1^2 + x2^2 <= 1.0 wp(x1,x2) else 0.0 end
zw = @. hw(x1', x2)
p = contour(x1, x2, zw, level=[1], color=[:white,:gray], levels=1, cbar=false, grid=false, fill=true)

# Sample the image set f(S)
N = 10^5
X = randn(2, N)
X = mapslices(c -> rand(1)[1]*c/sqrt(sum(c.^2)), X, dims=1)
f1 = mapslices(c->f[1](c), X, dims=1)[1, :]
f2 = mapslices(c->f[2](c), X, dims=1)[1, :]
scatter!(p, f1, f2, mc=:black, legend=false)

# Draw the unit circle
t = range(0, 2*pi, length=100)
xt = cos.(t)
yt = sin.(t)
plot!(p, xt, yt, color=:black, legend=false, ylimits=(-1,1), xlimits=(-1,1), aspect_ratio=:equal)