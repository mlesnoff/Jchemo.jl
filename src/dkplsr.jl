"""
    dkplsr(X, Y, weights = ones(nro(X)); nlv, 
        kern = :krbf, scal::Bool = false, kwargs...)
    dkplsr!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv, 
        kern = :krbf, scal = scal, kwargs...)
Direct kernel partial least squares regression (DKPLSR) (Bennett & Embrechts 2003).

* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. latent variables (LVs) to consider. 
* 'kern' : Type of kernel used to compute the Gram matrices.
    Possible values are :krbf of :kpol (see respective functions `krbf` and `kpol`).
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.
* `kwargs` : Named arguments to pass in the kernel function.

The method builds kernel Gram matrices and then runs a usual PLSR algorithm on them. 
This is faster (but not equivalent) to the "true" Nipals KPLSR algorithm described 
in Rosipal & Trejo (2001).

## References 
Bennett, K.P., Embrechts, M.J., 2003. An optimization perspective on kernel partial 
least squares regression, in: Advances in Learning Theory: Methods, Models and Applications, 
NATO Science Series III: Computer & Systems Sciences. IOS Press Amsterdam, pp. 227-250.

Rosipal, R., Trejo, L.J., 2001. Kernel Partial Least Squares Regression in 
Reproducing Kernel Hilbert Space. Journal of Machine Learning Research 2, 97-123.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
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

zcoef = Jchemo.coef(fm)
zcoef.int
zcoef.B
Jchemo.coef(fm; nlv = 7).B

Jchemo.transform(fm, Xtest)
Jchemo.transform(fm, Xtest; nlv = 7)

res = Jchemo.predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)

res = Jchemo.predict(fm, Xtest; nlv = 1:2)
res.pred[1]
res.pred[2]

fm = dkplsr(Xtrain, ytrain; nlv = nlv, kern = :kpol, degree = 2, gamma = 1e-1, coef0 = 10) ;
res = Jchemo.predict(fm, Xtest)
rmsep(res.pred, ytest)
plotxy(pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f    

# Example of fitting the function sinc(x)
# described in Rosipal & Trejo 2001 p. 105-106 

x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
fm = dkplsr(x, y; nlv = 2) ;
pred = Jchemo.predict(fm, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "ted model")
axislegend("Method")
f
```
""" 
function dkplsr(X, Y; par = Par())
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    dkplsr(X, Y, weights; par)
end

function dkplsr(X, Y, weights::Weight; par = Par())
    dkplsr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; par)
end

function dkplsr!(X::Matrix, Y::Matrix, weights::Weight; 
        par = Par())
    Q = eltype(X)
    p = nco(X)
    q = nco(Y)
    xscales = ones(Q, p)
    yscales = ones(eltype(Y), q)
    if par.scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        scale!(X, xscales)
        scale!(Y, yscales)
    end
    fkern = eval(Meta.parse(string("Jchemo.", par.kern)))
    K = fkern(X, X; par)     
    fm = plskern!(K, Y, weights; par)
    Dkplsr(X, fm, K, xscales, yscales, par)
end

""" 
    transform(object::Dkplsr, X; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transform(object::Dkplsr, X; nlv = nothing)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(scale(X, object.xscales), object.X; par = object.par)
    transform(object.fm, K; nlv = nlv)
end

"""
    coef(object::Dkplsr; nlv = nothing)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
   
""" 
function coef(object::Dkplsr; nlv = nothing)
    coef(object.fm; nlv = nlv)
end

"""
    predict(object::Dkplsr, X; nlv = nothing)
Compute Y-predictions from a fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
   
""" 
function predict(object::Dkplsr, X; nlv = nothing)
    a = nco(object.fm.T)
    isnothing(nlv) ? nlv = a : nlv = (max(0, minimum(nlv)):min(a, maximum(nlv)))
    le_nlv = length(nlv)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(scale(X, object.xscales), object.X; par = object.par)
    pred = predict(object.fm, K; nlv = nlv).pred
    if le_nlv == 1
        pred .= pred * Diagonal(object.yscales)
    else
        for i = 1:le_nlv
            pred[i] .= pred[i] * Diagonal(object.yscales)
        end
    end
    (pred = pred,)
end
