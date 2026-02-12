""" 
    plsravg(; kwargs...)
    plsravg(X, Y; kwargs...)
    plsravg(X, Y, weights::Weight; kwargs...)
    plsravg!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Averaging PLSR models with different numbers of  latent variables (PLSR-AVG).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g., function `mweight`).
Keyword arguments:
* `nlv` : A range of nb. of latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected standard deviation.

Ensemblist method where the predictions are computed by averaging the predictions of a set of models built 
with different numbers of LVs.

For instance, if argument `nlv` is set to `nlv` = `5:10`, the prediction for a new observation is the simple average
of the predictions returned by the models with 5 LVs, 6 LVs, ... 10 LVs, respectively.

## References
Lesnoff, M., Andueza, D., Barotin, C., Barre, V., Bonnal, L., Fernández Pierna, J.A., Picard, F., Vermeulen, V., 
Roger, J.-M., 2022. Averaging and Stacking Partial Least Squares Regression Models to Predict the Chemical Compositions 
and the Nutritive Values of Forages from Spectral Near Infrared Data. Applied Sciences 12, 7850. 
https://doi.org/10.3390/app12157850

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2") 
@load db dat
@names dat
X = dat.X 
Y = dat.Y
@head Y
y = Y.ndf
#y = Y.dm
n = nro(X)
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(y, s)
Xtest = X[s, :]
ytest = y[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = n, ntrain, ntest)

nlv = 0:30
#nlv = 5:20
#nlv = 25
model = plsravg(; nlv) ;
fit!(model, Xtrain, ytrain)

res = predict(model, Xtest)
@head res.pred
res.predlv   # predictions for each nb. of LVs 
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",  
    ylabel = "Observed").f   
```
""" 
plsravg(; kwargs...) = JchemoModel(plsravg, nothing, kwargs)

function plsravg(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plsravg(X, Y, weights; kwargs...)
end

function plsravg(X, Y, weights::Weight; kwargs...)
    plsravg!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function plsravg!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParPlsr, kwargs).par
    algo = plsravg_unif!
    fitm = algo(X, Y, weights; kwargs...)
    Plsravg(fitm, par) 
end

"""
    predict(object::Plsravg, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Plsravg, X)
    res = predict(object.fitm, X)
    (pred = res.pred, predlv = res.predlv)
end

## Note: There is no 'transf' nor 'coef' functions for Plsravg.

