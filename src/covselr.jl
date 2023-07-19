struct Covselr2
    fm 
    fm_covsel
end

"""
    covselr(X, Y, weights = ones(nro(X)); nlv,
        scal::Bool = false)
MLR on variables selected from partial covariance (Covsel).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. variables to select.
* `scal` : Boolean. If `true`, each column of `X`
    is scaled by its uncorrected standard deviation.

A number of `nlv` variables (X-columns) are selected with the Covsel method
(see function `covsel`), and then a MLR is implemened on these variables. 

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
  
nlv = 15
fm = covselr(Xtrain, ytrain; nlv = nlv) ;
pnames(fm)
Jchemo.coef(fm.fm)

res = Jchemo.predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
plotxy(vec(res.pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f    
```
""" 
function covselr(X, Y; nlv,
        scal::Bool = false)
    fm0 = covsel(X, Y; nlv = nlv,
        scal = scal)
    fm = mlr(vcol(X, fm0.sel), Y)
    Covselr2(fm, fm0)
end 

"""
    predict(object::Covselr, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Covselr2, X)
    pred = predict(object.fm, X[:, object.fm_covsel.sel]).pred
    (pred = pred,)
end


