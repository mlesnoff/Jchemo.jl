"""
    mlr(; kwargs...)
    mlr(X, Y; kwargs...)
    mlr(X, Y, weights::Weight; kwargs...)
    mlr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Compute a mutiple linear regression model (MLR) by using the QR algorithm.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `noint` : Boolean. Define if the model is computed with an intercept or not.

Safe but can be little slower than other methods.

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
#model = mlrchol()
#model = mlrpinv()
#model = mlrpinvn() 
fit!(model, Xtrain, ytrain) 
@names model
@names model.fitm
fitm = model.fitm ;
fitm.B
fitm.int 
coef(model) 
res = predict(model, Xtest)
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",  
    ylabel = "Observed").f    

model = mlr(noint = true)
fit!(model, Xtrain, ytrain) 
coef(model) 

model = mlrvec()
fit!(model, Xtrain[:, 1], ytrain) 
coef(model) 
```
""" 
mlr(; kwargs...) = JchemoModel(mlr, nothing, kwargs)

function mlr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    mlr(X, Y, weights; kwargs...)
end

function mlr(X, Y, weights::Weight; kwargs...)
    mlr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function mlr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
    par = recovkw(ParMlr, kwargs).par
    Q = eltype(X)
    isa(Y, BitMatrix) ? Y = convert.(Q, Y) : nothing
    sqrtw = sqrt.(weights.w)
    if par.noint
        q = nco(Y)
        fweight!(X, sqrtw)
        fweight!(Y, sqrtw)
        B = X \ Y
        int = zeros(q)'
    else
        xmeans = colmean(X, weights) 
        ymeans = colmean(Y, weights)   
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
        fweight!(X, sqrtw)
        fweight!(Y, sqrtw)
        B = X \ Y
        int = ymeans' .- xmeans' * B
    end
    Mlr(B, int, weights, par)
end

"""
    mlrchol()
    mlrchol(X, Y)
    mlrchol(X, Y, weights::Weight)
    mlrchol!mlrchol!(X::Matrix, Y::Matrix, weights::Weight)
Compute a mutiple linear regression model (MLR) using the Normal equations and a Choleski factorization.
* `X` : X-data, with nb. columns >= 2 (required by function cholesky).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`). 

Only compute a model with intercept.

Faster but can be less accurate (based on squared element X'X).

See function `mlr` for examples.
""" 
mlrchol(; kwargs...) = JchemoModel(mlrchol, nothing, kwargs)

function mlrchol(X, Y)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    mlrchol(X, Y, weights)
end

function mlrchol(X, Y, weights::Weight)
    mlrchol!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights)
end

function mlrchol!(X::Matrix, Y::Matrix, weights::Weight)
    @assert nco(X) > 1 "The Method only works for X with nb. columns > 1."
    sqrtw = sqrt.(weights.w)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    fcenter!(X, xmeans)
    fcenter!(Y, ymeans)
    fweight!(X, sqrtw)
    fweight!(Y, sqrtw)
    B = cholesky!(Hermitian(X' * X)) \ (X' * Y)
    int = ymeans' .- xmeans' * B
    MlrNoArg(B, int, weights)
end

"""
    mlrpinv()
    mlrpinv(X, Y; kwargs...)
    mlrpinv(X, Y, weights::Weight; kwargs...)
    mlrpinv!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Compute a mutiple linear regression model (MLR)  by using 
    a pseudo-inverse. 
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments:
* `noint` : Boolean. Define if the model is computed with an intercept or not.

Safe but can be slower.  

See function `mlr` for examples.
""" 
mlrpinv(; kwargs...) = JchemoModel(mlrpinv, nothing, kwargs)

function mlrpinv(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    mlrpinv(X, Y, weights; kwargs...)
end

function mlrpinv(X, Y, weights::Weight; kwargs...)
    mlrpinv!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function mlrpinv!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParMlr, kwargs).par
    sqrtw = sqrt.(weights.w)
    if par.noint
        q = nco(Y)
        fweight!(X, sqrtw)
        fweight!(Y, sqrtw)
        tol = sqrt(eps(real(float(one(eltype(X))))))      # see ?pinv
        B = pinv(X, rtol = tol) * Y
        int = zeros(q)'
    else
        xmeans = colmean(X, weights) 
        ymeans = colmean(Y, weights)   
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
        fweight!(X, sqrtw)
        fweight!(Y, sqrtw)
        tol = sqrt(eps(real(float(one(eltype(X))))))      # see ?pinv
        B = pinv(X, rtol = tol) * Y
        int = ymeans' .- xmeans' * B
    end
    Mlr(B, int, weights, par)
end

"""
    mlrpinvn() 
    mlrpinvn(X, Y)
    mlrpinvn(X, Y, weights::Weight)
    mlrpinvn!mlrchol!(X::Matrix, Y::Matrix, weights::Weight)
Compute a mutiple linear regression model (MLR) by using the Normal equations and a pseudo-inverse.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`). 

Safe and fast for p not too large.

Only compute a model with intercept.

See function `mlr` for examples.
""" 
mlrpinvn(; kwargs...) = JchemoModel(mlrpinvn, nothing, kwargs)

function mlrpinvn(X, Y)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    mlrpinvn(X, Y, weights)
end

function mlrpinvn(X, Y, weights::Weight)
    mlrpinvn!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights)
end

function mlrpinvn!(X::Matrix, Y::Matrix, weights::Weight)
    sqrtw = sqrt.(weights.w)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    fcenter!(X, xmeans)
    fcenter!(Y, ymeans)
    fweight!(X, sqrtw)
    fweight!(Y, sqrtw)
    XtX = X' * X
    tol = sqrt(eps(real(float(one(eltype(XtX))))))
    B = pinv(XtX, rtol = tol) * (X' * Y)
    int = ymeans' .- xmeans' * B
    MlrNoArg(B, int, weights)
end

"""
    mlrvec(; kwargs...)
    mlrvec(X, Y; kwargs...)
    mlrvec(X, Y, weights::Weight; kwargs...)
    mlrvec!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Compute a simple (univariate x) linear regression model.
* `x` : Univariate X-data (n).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments:
* `noint` : Boolean. Define if the model is computed with an intercept or not.

See function `mlr` for examples.
""" 
mlrvec(; kwargs...) = JchemoModel(mlrvec, nothing, kwargs)

function mlrvec(x, Y; kwargs...)
    Q = eltype(x[1, 1])
    weights = mweight(ones(Q, nro(x)))
    mlrvec(x, Y, weights; kwargs...)
end

function mlrvec(x, Y, weights::Weight; kwargs...)
    mlrvec!(copy(ensure_mat(x)), copy(ensure_mat(Y)), weights; kwargs...)
end

function mlrvec!(x::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParMlr, kwargs).par
    @assert nco(x) == 1 "Method only working for univariate x."
    sqrtw = sqrt.(weights.w)
    if par.noint
        q = nco(Y)
        fweight!(x, sqrtw)
        fweight!(Y, sqrtw)
        B = x' * Y / dot(x, x)
        int = zeros(q)'
    else
        xmeans = colmean(x, weights) 
        ymeans = colmean(Y, weights)   
        fcenter!(x, xmeans)
        fcenter!(Y, ymeans)
        fweight!(x, sqrtw)
        fweight!(Y, sqrtw)
        B = x' * Y / dot(x, x)
        int = ymeans' .- xmeans' * B
    end
    Mlr(B, int, weights, par)
end

"""
    coef(object::Union{Mlr, MlrNoArg, Rrchol})
Compute the coefficients of the fitted model.
* `object` : The fitted model.
""" 
function coef(object::Union{Mlr, MlrNoArg, Rrchol})
    (B = object.B, int = object.int)
end

"""
    predict(object::Union{Mlr, MlrNoArg, Rrchol}, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Union{Mlr, MlrNoArg, Rrchol}, X)
    X = ensure_mat(X)
    z = coef(object)
    pred = z.int .+ X * z.B
    (pred = pred,)
end



