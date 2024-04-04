"""
    mlrda(; kwargs...)
    mlrda(X, y; kwargs...)
    mlrda(X, y, weights::Weight)
Discrimination based on multple linear regression 
    (MLR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments:
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (the vector must be sorted in the same order as `mlev(x)`).

The training variable `y` (univariate class membership) is 
transformed to a dummy table (Ydummy) containing nlev columns, 
where nlev is the number of classes present in `y`. Each column of 
Ydummy is a dummy (0/1) variable. Then, a multiple linear regression 
(MLR) is run on {`X`, Ydummy}, returning predictions of the dummy 
variables (= object `posterior` returned by fuction `predict`).  
These predictions can be considered as unbounded estimates (i.e. 
eventuall outside of [0, 1]) of the class membership probabilities. 
For a given observation, the final prediction is the class 
corresponding to the dummy variable for which the probability 
estimate is the highest.

In the high-level version of the function, the observation weights used in 
the MLR are defined with argument `prior`. For other choices, use the 
low-level version (argument `weights`).

## Examples
```julia
using JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2")
@load db dat
pnames(dat)
@head dat.X
X = dat.X[:, 1:4]
y = dat.X[:, 5]
n = nro(X)
ntest = 30
s = samprand(n, ntest)
Xtrain = X[s.train, :]
ytrain = y[s.train]
Xtest = X[s.test, :]
ytest = y[s.test]
ntrain = n - ntest
(ntot = n, ntrain, ntest)
tab(ytrain)
tab(ytest)

mod = mlrda()
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
fm.lev
fm.ni

res = predict(mod, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt
```
""" 
function mlrda(X, y; kwargs...)
    par = recovkwargs(Par, kwargs)
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    mlrda(X, y, weights)
end

function mlrda(X, y, weights::Weight)
    X = ensure_mat(X)
    y = ensure_mat(y)
    res = dummy(y)
    ni = tab(y).vals
    fm = mlr(X, res.Y, weights)
    Mlrda(fm, res.lev, ni)
end

"""
    predict(object::Mlrda, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Mlrda, X)
    X = ensure_mat(X)
    m = nro(X)
    zp = predict(object.fm, X).pred
    z =  mapslices(argmax, zp; dims = 2) 
    pred = reshape(replacebylev2(z, object.lev), m, 1)
    (pred = pred, posterior = zp)
end
    
