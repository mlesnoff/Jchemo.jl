"""
    krrda(X, y, weights = ones(nro(X)); lb, 
        scal = scal, kern = :krbf, kwargs...)
Discrimination based on kernel ridge regression (KRR-DA).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `lb` : Ridge regularization parameter "lambda".
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* Other arguments to pass in the kernel: See `?kplsr`.

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
using JchemoData
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2") 
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

gamma = .01
lb = .001
fm = krrda(Xtrain, ytrain; lb = lb, gamma = gamma) ;    
pnames(fm)
pnames(fm.fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)
confusion(res.pred, ytest).cnt

Jchemo.predict(fm, Xtest; lb = [.1; .01]).pred
```
""" 
function krrda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    krrda(X, y, weights; kwargs...)
end

function krrda(X, y, weights::Weight; 
        kwargs...)  
    res = dummy(y)
    ni = tab(y).vals
    fm = krr(X, res.Y, weights; 
        kwargs...)
    Krrda(fm, res.lev, ni, kwargs, par)
end


