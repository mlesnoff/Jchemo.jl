"""
    krrda(X, y, weights = ones(nro(X)); lb, 
        scal = scal, kern = "krbf", kwargs...)
Discrimination based on kernel ridge regression (KRR-DA).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `lb` : A value of the regularization parameter "lambda".
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* Other arguments: see '?kplsr'.

The training variable `y` (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy (0/1) variable. 
Then, a RR is implemented on the `y` and each column of Ydummy,
returning predictions of the dummy variables (= object `posterior` returned by 
function `predict`). 
These predictions can be considered as unbounded 
estimates (i.e. eventually outside of [0, 1]) of the class membership probabilities.
For a given observation, the final prediction is the class corresponding 
to the dummy variable for which the probability estimate is the highest.

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

gamma = .01
lb = .001
fm = krrda(Xtrain, ytrain; lb = lb, gamma = gamma) ;    
pnames(fm)
pnames(fm.fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)

Jchemo.predict(fm, Xtest; lb = [.1; .01]).pred
```
""" 
function krrda(X, y, weights = ones(nro(X)); lb, 
        kern = "krbf", scal = false, kwargs...)
    z = dummy(y)
    fm = krr(X, z.Y, weights; lb = lb, 
        kern = kern, scal = scal, kwargs...)
    Rrda(fm, z.lev, z.ni)
end


