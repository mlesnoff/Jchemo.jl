"""
    mlrda(; kwargs...)
    mlrda(X, y; kwargs...)
    mlrda(X, y, weights::Weight)
Discrimination based on multple linear regression (MLR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments:
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).

The method is as follows:

1) The training variable `y` (univariate class membership) is 
    transformed to a dummy table (Ydummy) containing nlev columns, 
    where nlev is the number of classes present in `y`. Each column of 
    Ydummy is a dummy (0/1) variable. 
2) Then, a multiple linear regression (MLR) is run on {`X`, Ydummy}, returning 
    predictions of the dummy variables (= object `posterior` returned by 
    fuction `predict`).  These predictions can be considered as unbounded estimates 
    (i.e. eventuall outside of [0, 1]) of the class membership probabilities.
3) For a given observation, the final prediction is the class 
    corresponding to the dummy variable for which the probability 
    estimate is the highest.

In the high-level version of the present functions, the observation 
weights are automatically defined by the given priors (argument `prior`): 
the sub-totals by class of the observation weights are set equal to the prior 
probabilities. The low-level version (argument `weights`) allows to implement 
other choices.

## Examples
```julia
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2")
@load db dat
@names dat
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

model = mlrda()
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm
fitm = model.fitm ;
fitm.lev
fitm.ni

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt
```
""" 
mlrda(; kwargs...) = JchemoModel(mlrda, nothing, kwargs)

function mlrda(X, y; kwargs...)
    par = recovkw(ParMlrda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    mlrda(X, y, weights; kwargs...)
end

function mlrda(X, y, weights::Weight; kwargs...)
    par = recovkw(ParMlrda, kwargs).par
    X = ensure_mat(X)
    y = ensure_mat(y)
    res = dummy(y)
    ni = tab(y).vals
    fitm = mlr(X, res.Y, weights)
    Mlrda(fitm, res.lev, ni, par)
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
    zp = predict(object.fitm, X).pred
    z =  mapslices(argmax, zp; dims = 2) 
    pred = reshape(recod_indbylev(z, object.lev), m, 1)
    (pred = pred, posterior = zp)
end
    
