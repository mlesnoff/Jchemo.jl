struct Mlrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

"""
    mlrda(X, y, weights = ones(size(X, 1)))
Discrimination based on multple linear regression (MLR-DA).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations.

The training variable `y` (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy (0/1) variable. 
Then, a multiple linear regression (MLR) is run between the `X` and and each column 
of Ydummy, returning predictions of the dummy variables (= object `posterior` 
returned by fuction `predict`).  
These predictions can be  considered as unbounded 
estimates (i.e. eventuall outside of [0, 1]) of the class membership probabilities.
For a given observation, the final prediction is the class corresponding 
to the dummy variable for which the probability estimate is the highest.

## Examples
```julia
using JchemoData, JLD2, StatsBase
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "iris.jld2") 
@load db dat
pnames(dat)
summ(dat.X)

X = dat.X[:, 1:4] 
y = dat.X[:, 5]
n = nro(X)

ntrain = 120
s = sample(1:n, ntrain; replace = false) 
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

tab(ytrain)
tab(ytest)

fm = mlrda(Xtrain, ytrain) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.posterior
res.pred
err(res.pred, ytest)
```
""" 
function mlrda(X, y, weights = ones(size(X, 1)))
    z = dummy(y)
    fm = mlr(X, z.Y, weights)
    Mlrda(fm, z.lev, z.ni)
end

"""
    predict(object::Mlrda, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Mlrda, X)
    X = ensure_mat(X)
    m = size(X, 1)
    zp = predict(object.fm, X).pred
    z =  mapslices(argmax, zp; dims = 2) 
    pred = reshape(replacebylev2(z, object.lev), m, 1)
    (pred = pred, posterior = zp)
end
    


