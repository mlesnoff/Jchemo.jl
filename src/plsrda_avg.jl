struct PlsdaAvg  # for plsrda_avg, plsrla_avg and plsqda_avg 
    fm
    nlv
    w_mod
    lev::AbstractVector
    ni::AbstractVector
end

""" 
    plsrda_avg(X, y, weights = ones(size(X, 1)); nlv)
Averaging of PLSR-DA models with different numbers of LVs.
* `X` : X-data.
* `y` : y-data (class membership).
* weights : Weights of the observations.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 
    are averaged). Syntax such as "10" is also allowed ("10": correponds to
    the single model with 10 LVs).

Ensemblist method where the predictions are calculated by "averaging" 
the predictions of a set of models built with different numbers of 
latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the most occurent class within the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.

## Examples
```julia
using JLD2, CairoMakie
mypath = dirname(dirname(pathof(Jchemo)))
db = joinpath(mypath, "data", "forages.jld2") 
@load db dat
pnames(dat)

X = dat.X 
Y = dat.Y 
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]

tab(ytrain)
tab(ytest)

nlv = "0:40"
fm = plsrda_avg(Xtrain, ytrain; nlv = nlv) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)
```
""" 
function plsrda_avg(X, y, weights = ones(size(X, 1)); nlv)
    n = size(X, 1)
    p = size(X, 2)
    nlv = eval(Meta.parse(nlv))
    nlvmax = maximum(nlv)
    nlv = (max(minimum(nlv), 0):min(nlvmax, n, p))
    w = ones(nlvmax + 1)
    w_mod = mweight(w[collect(nlv) .+ 1])   # uniform weights for the models
    fm = plsrda(X, y, weights; nlv = nlvmax)
    PlsdaAvg(fm, nlv, w_mod, fm.lev, fm.ni)
end

""" 
    plslda_avg(X, y, weights = ones(size(X, 1)); nlv)
Averaging of PLS-LDA models with different numbers of LVs.
* `X` : X-data.
* `y` : y-data (class membership).
* weights : Weights of the observations.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 
    are averaged). Syntax such as "10" is also allowed ("10": correponds to
    the single model with 10 LVs).

Ensemblist method where the predictions are calculated by "averaging" 
the predictions of a set of models built with different numbers of 
latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the most occurent class within the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(Jchemo)))
db = joinpath(mypath, "data", "forages.jld2") 
@load db dat
pnames(dat)

Xtrain = dat.Xtrain
ytrain = dat.Ytrain.y
Xtest = dat.Xtest
ytest = dat.Ytest.y

tab(ytrain)
tab(ytest)

fm = plslda_avg(Xtrain, ytrain; nlv = "1:40") ;    # minimum of nlv must be >=1 (conversely to plsrda_avg)
#fm = plslda_avg(Xtrain, ytrain; nlv = "1:20") ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)
```
""" 
function plslda_avg(X, y, weights = ones(size(X, 1)); nlv)
    n = size(X, 1)
    p = size(X, 2)
    nlv = eval(Meta.parse(nlv))
    nlvmax = maximum(nlv)
    nlv = (max(minimum(nlv), 0):min(nlvmax, n, p))
    w = ones(nlvmax + 1)
    w_mod = mweight(w[collect(nlv) .+ 1])   # uniform weights for the models
    fm = plslda(X, y, weights; nlv = nlvmax)
    PlsdaAvg(fm, nlv, w_mod, fm.lev, fm.ni)
end

""" 
    plsqda_avg(X, y, weights = ones(size(X, 1)); nlv)
Averaging of PLS-QDA models with different numbers of LVs.
* `X` : X-data.
* `y` : y-data (class membership).
* weights : Weights of the observations.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 
    are averaged). Syntax such as "10" is also allowed ("10": correponds to
    the single model with 10 LVs).

Ensemblist method where the predictions are calculated by "averaging" 
the predictions of a set of models built with different numbers of 
latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the most occurent class within the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.

See `?plslda_avg` for examples.
""" 
function plsqda_avg(X, y, weights = ones(size(X, 1)); nlv)
    n = size(X, 1)
    p = size(X, 2)
    nlv = eval(Meta.parse(nlv))
    nlvmax = maximum(nlv)
    nlv = (max(minimum(nlv), 0):min(nlvmax, n, p))
    w = ones(nlvmax + 1)
    w_mod = mweight(w[collect(nlv) .+ 1])   # uniform weights for the models
    fm = plsqda(X, y, weights; nlv = nlvmax)
    PlsdaAvg(fm, nlv, w_mod, fm.lev, fm.ni)
end

"""
    predict(object::PlsdaAvg, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::PlsdaAvg, X)
    X = ensure_mat(X)
    m = size(X, 1)
    nlv = object.nlv
    le_nlv = length(nlv)
    zpred = predict(object.fm, X; nlv = nlv).pred
    if(le_nlv == 1)
        pred = zpred
    else
        z = reduce(hcat, zpred)
        pred = similar(object.fm.lev, m, 1)
        @inbounds for i = 1:m    
            pred[i, :] .= findmax_cla(z[i, :], object.w_mod)
        end
    end
    (pred = pred, predlv = zpred)
end


