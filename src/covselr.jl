struct Covselr
    fm 
    sel::DataFrame
    cov2::Vector{Float64}
end

"""
    covselr(X, Y; nlv = nothing, , typ = "corr")
MLR on variables selected from partial correlation or covariance (Covsel).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `nlv` : Nb. variables to select.
* `typ` : Criterion used at each selection in Covsel (See `?covsel`.). 

A number of `nlv` variables (X-columns) are selected with the Covsel method
function `covsel`), and then a MLR is implemened on these variables. 

## Examples
```julia
using JLD2, CairoMakie
mypath = dirname(dirname(pathof(Jchemo)))
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
  
nlv = 15
fm = covselr(Xtrain, ytrain; nlv = nlv) ;
pnames(fm)
coef(fm.fm)

res = predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
f, ax = scatter(vec(res.pred), ytest)
abline!(ax, 0, 1)
f
```
""" 
function covselr(X, Y; nlv, typ = "corr")
    res = covsel(X, Y; nlv = nlv, typ = typ)
    zX = vcol(X, res.sel.sel)
    fm = mlr(zX, Y)
    Covselr(fm, res.sel, res.cov2)
end 

function predict(object::Covselr, X)
    pred = predict(object.fm, X[:, object.sel.sel]).pred
    (pred = pred,)
end




