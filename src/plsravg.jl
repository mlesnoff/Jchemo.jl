""" 
    plsravg(; kwargs...)
    plsravg(X, Y; kwargs...)
    plsravg(X, Y, weights::ProbabilityWeights; kwargs...)
    plsravg!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
Averaging PLSR models with different numbers of  latent variables (PLSR-AVG).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : A range of nb. of latent variables (LVs) to compute.
* `scal` : Symbol defining the column scaling of `X` and `Y`. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

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
#nlv = 25:25
model = plsravg(; nlv) ;
fit!(model, Xtrain, ytrain)

res = predict(model, Xtest)
@names res
@head res.pred
res.predlv   # predictions for each nb. of LVs 
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",  
    ylabel = "Observed").f   
```
""" 
plsravg(; kwargs...) = JchemoModel(plsravg, nothing, kwargs)

function plsravg(X, Y; kwargs...)
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    plsravg(X, Y, weights; kwargs...)
end

function plsravg(X, Y, weights::ProbabilityWeights; kwargs...)
    plsravg!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function plsravg!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    par = recovkw(ParPlsravg{Q}, kwargs).par
    if par.algo == :unif
        fitm = plsravg_unif!(X, Y, weights; kwargs...)
    end
    Plsravg(fitm, par)  
end

## Note: There is no 'transf' nor 'coef' functions for Plsravg.

"""
    predict(object::Plsravg, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Plsravg, X)
    #res = predict(object.fitm, X)
    #(pred = res.pred, predlv = res.predlv, nlv = res.nlv)
    predict(object.fitm, X)
end


