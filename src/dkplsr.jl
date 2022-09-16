struct Dkplsr
    X::Array{Float64}
    fm
    K::Array{Float64}
    kern
    dots
end

"""
    dkplsr(X, Y, weights = ones(size(X, 1)); nlv , kern = "krbf", kwargs...)
    dkplsr!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); nlv, kern = "krbf", kwargs...)
Direct kernel partial least squares regression (DKPLSR) (Bennett & Embrechts 2003).

* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations.
* `nlv` : Nb. latent variables (LVs) to consider. 
* 'kern' : Type of kernel used to compute the Gram matrices.
    Possible values are "krbf" of "kpol" (see respective functions `krbf` and `kpol`).
* `kwargs` : Named arguments to pass in the kernel function.

The method builds kernel Gram matrices and then runs a usual PLSR algorithm on them. This is faster 
(but not equivalent) to the "true" NIPALS KPLSR algorithm described in Rosipal & Trejo (2001).

## References 
Bennett, K.P., Embrechts, M.J., 2003. An optimization perspective on kernel partial least squares regression, 
in: Advances in Learning Theory: Methods, Models and Applications, 
NATO Science Series III: Computer & Systems Sciences. IOS Press Amsterdam, pp. 227-250.

Rosipal, R., Trejo, L.J., 2001. Kernel Partial Least Squares Regression in Reproducing Kernel Hilbert Space. 
Journal of Machine Learning Research 2, 97-123.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 20 ; gamma = 1e-1
fm = dkplsr(Xtrain, ytrain; nlv = nlv, gamma = gamma) ;
fm.fm.T

zcoef = coef(fm)
zcoef.int
zcoef.B
coef(fm; nlv = 7).B

transform(fm, Xtest)
transform(fm, Xtest; nlv = 7)

res = predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)

res = predict(fm, Xtest; nlv = 1:2)
res.pred[1]
res.pred[2]

fm = dkplsr(Xtrain, ytrain; nlv = nlv, kern = "kpol", degree = 2, gamma = 1e-1, coef0 = 10) ;
res = predict(fm, Xtest)
rmsep(res.pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f    

# Example of fitting the function sinc(x)
# described in Rosipal & Trejo 2001 p. 105-106 

x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
fm = dkplsr(x, y; nlv = 2) ;
pred = predict(fm, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
function dkplsr(X, Y, weights = ones(size(X, 1)); nlv, kern = "krbf", kwargs...)
    dkplsr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; 
        nlv = nlv, kern = kern, kwargs...)
end

function dkplsr!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); 
        nlv, kern = "krbf", kwargs...)
    fkern = eval(Meta.parse(kern))
    K = fkern(X, X; kwargs...)     # In the future: fkern!(K, X, X; kwargs...)
    fm = plskern!(K, Y; nlv = nlv)
    Dkplsr(X, fm, K, kern, kwargs)
end

""" 
    transform(object::Dkplsr, X; nlv = nothing)
Compute LVs (score matrix "T") from a fitted model and X-data.
* `object` : The maximal fitted model.
* `X` : X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider. If nothing, it is the maximum nb. LVs.
""" 
function transform(object::Dkplsr, X; nlv = nothing)
    fkern = eval(Meta.parse(object.kern))
    K = fkern(X, object.X; object.dots...)
    transform(object.fm, K; nlv = nlv)
end

"""
    coef(object::Dkplsr; nlv = nothing)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
    If nothing, it is the maximum nb. LVs.
""" 
function coef(object::Dkplsr; nlv = nothing)
    coef(object.fm; nlv = nlv)
end

"""
    predict(object::Dkplsr, X; nlv = nothing)
Compute Y-predictions from a fitted model and X-data.
* `object` : The maximal fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
    If nothing, it is the maximum nb. LVs.
""" 
function predict(object::Dkplsr, X; nlv = nothing)
    fkern = eval(Meta.parse(object.kern))
    K = fkern(X, object.X; object.dots...)
    pred = predict(object.fm, K; nlv = nlv).pred
    (pred = pred,)
end
