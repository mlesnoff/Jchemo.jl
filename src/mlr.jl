"""
    mlr(; kwargs...)
    mlr(X, Y; kwargs...)
    mlr(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    mlr!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
Mutiple linear regression model (MLR).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `noint` : Boolean. Define if the model is computed with an intercept or not (default to `false`).

Use the matrix division operator (polyalgorithm, see the related help).

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie 
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
@names dat
@head dat.X
X = dat.X[:, 2:4]
y = dat.X[:, 1]
ntot = nro(X)
ntest = 30
s = samprand(ntot, ntest) 
Xtrain = X[s.train, :]
ytrain = y[s.train]
Xtest = X[s.test, :]
ytest = y[s.test]

model = mlr()
fit!(model, Xtrain, ytrain) 
@names model
fitm = model.fitm ;
@names fitm

coef(model) 
fitm.B
fitm.int 

res = predict(model, Xtest)
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",  
    ylabel = "Observed").f    

model = mlr(noint = true)
fit!(model, Xtrain, ytrain) 
coef(model) 
```
""" 
mlr(; kwargs...) = JchemoModel(mlr, nothing, kwargs)

function mlr(X, Y; kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = pweight(ones(eltype(X), nro(X)))
    mlr(X, Y, weights; kwargs...)
end

function mlr(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    mlr!(copy(X), copy(Y), weights; kwargs...)
end

function mlr!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    par = recovkw(ParMlr, kwargs).par
    sqrtw = sqrt.(weights.values)
    if par.noint
        q = nco(Y)
        fweightr!(X, sqrtw)
        fweightr!(Y, sqrtw)
        B = X \ Y
        int = zeros(q)'
    else
        xmeans = colmean(X, weights) 
        ymeans = colmean(Y, weights)   
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
        fweightr!(X, sqrtw)
        fweightr!(Y, sqrtw)
        B = X \ Y
        int = ymeans' .- xmeans' * B
    end
    Mlr(B, int, weights, par)
end

"""
    coef(object::Mlr)
Compute the coefficients of the fitted model.
* `object` : The fitted model.
""" 
function coef(object::Mlr)
    (B = object.B, int = object.int)
end

"""
    predict(object::Mlr, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Mlr, X)
    X = ensure_mat(X)
    z = coef(object)
    pred = z.int .+ X * z.B
    (pred = pred,)
end



