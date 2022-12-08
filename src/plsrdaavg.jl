struct Plsdaavg  # for plsrdaavg, plsrlaavg and plsqdaavg 
    fm
    nlv
    w_mod
    lev::AbstractVector
    ni::AbstractVector
end

""" 
    plsrdaavg(X, y, weights = ones(nro(X)); nlv)
Averaging of PLSR-DA models with different numbers of 
    latent variables (LVs).
* `X` : X-data.
* `y` : y-data (class membership).
* weights : Weights of the observations.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 
    are averaged). Syntax such as "10" is also allowed ("10": correponds to
    the single model with 10 LVs).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Ensemblist method where the predictions are calculated by "averaging" 
the predictions of a set of models built with different numbers of LVs.

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the most occurent class within the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
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
fm = plsrdaavg(Xtrain, ytrain; nlv = nlv) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)
```
""" 
function plsrdaavg(X, y, weights = ones(nro(X)); nlv,
    scal = false)
    n, p = size(X)
    nlv = eval(Meta.parse(nlv))
    nlvmax = maximum(nlv)
    nlv = (max(minimum(nlv), 0):min(nlvmax, n, p))
    w = ones(nlvmax + 1)
    # Uniform weights for the models
    w_mod = mweight(w[collect(nlv) .+ 1])
    # End
    fm = plsrda(X, y, weights; nlv = nlvmax,
        scal = scal)
    Plsdaavg(fm, nlv, w_mod, fm.lev, fm.ni)
end

"""
    predict(object::Plsdaavg, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Plsdaavg, X)
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


