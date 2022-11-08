""" 
    plslda_avg(X, y, weights = ones(nro(X)); nlv,
        scal = false)
Averaging of PLS-LDA models with different numbers of LVs.
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
the predictions of a set of models built with different numbers of 
latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the most occurent class within the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "forages.jld2") 
@load db dat
pnames(dat)

Xtrain = dat.Xtrain
ytrain = dat.Ytrain.y
Xtest = dat.Xtest
ytest = dat.Ytest.y

tab(ytrain)
tab(ytest)

# minimum of nlv must be >=1 (conversely to plsrda_avg)
fm = plslda_avg(Xtrain, ytrain; nlv = "1:40") ;    
#fm = plslda_avg(Xtrain, ytrain; nlv = "1:20") ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)
```
""" 
function plslda_avg(X, y, weights = ones(nro(X)); nlv,
        scal = false)
    n = size(X, 1)
    p = size(X, 2)
    nlv = eval(Meta.parse(nlv))
    nlvmax = maximum(nlv)
    nlv = (max(minimum(nlv), 0):min(nlvmax, n, p))
    w = ones(nlvmax + 1)
    # Uniform weights for the models
    w_mod = mweight(w[collect(nlv) .+ 1])
    # End
    fm = plslda(X, y, weights; nlv = nlvmax,
        scal = scal)
    PlsdaAvg(fm, nlv, w_mod, fm.lev, fm.ni)
end




