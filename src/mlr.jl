"""
    mlr(X, Y; kwargs...)
    mlr(X, Y, weights::Weight; kwargs...)
    mlr!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Compute a mutiple linear regression model (MLR) by using the QR algorithm.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `noint` : Boolean. Define if the model is computed 
    with an intercept or not.

Safe but can be little slower than other methods.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie 
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
pnames(dat)
@head dat.X
X = dat.X[:, 2:4]
y = dat.X[:, 1]
n = nro(X)
ntest = 30
s = samprand(n, ntest) 
Xtrain = X[s.train, :]
ytrain = y[s.train]
Xtest = X[s.test, :]
ytest = y[s.test]

model = mlr)
#model = mlrchol)
#model = mlrpinv)
#model = mlrpinvn) 
fit!(model, Xtrain, ytrain) 
pnames(model)
pnames(model.fitm)
fitm = model.fitm ;
fitm.B
fitm.int 
coef(model) 
res = predict(model, Xtest)
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, 
    xlabel = "Prediction", ylabel = "Observed").f    

model = mlr; noint = true)
fit!(model, Xtrain, ytrain) 
coef(model) 
```
""" 
function mlr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    mlr(X, Y, weights; kwargs...)
end

function mlr(X, Y, weights::Weight; kwargs...)
    mlr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function mlr!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParMlr, kwargs).par
    sqrtD = Diagonal(sqrt.(weights.w))
    if par.noint
        q = nco(Y)
        B = (sqrtD * X) \ (sqrtD * Y)
        int = zeros(q)'
    else
        xmeans = colmean(X, weights) 
        ymeans = colmean(Y, weights)   
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
        B = (sqrtD * X) \ (sqrtD * Y)
        int = ymeans' .- xmeans' * B
    end
    Mlr(B, int, weights, par)
end

"""
    mlrchol(X, Y)
    mlrchol(X, Y, weights::Weight)
    mlrchol!mlrchol!(X::Matrix, Y::Matrix, weights::Weight)
Compute a mutiple linear regression model (MLR) using the Normal equations 
    and a Choleski factorization.
* `X` : X-data, with nb. columns >= 2 (required by function cholesky).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 

Compute a model with intercept.

Faster but can be less accurate (based on squared element X'X).

See function `mlr` for examples.
""" 
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
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    fcenter!(X, xmeans)
    fcenter!(Y, ymeans)
    XtD = X' * Diagonal(weights.w)
    B = cholesky!(Hermitian(XtD * X)) \ (XtD * Y)
    int = ymeans' .- xmeans' * B
    MlrNoArg(B, int, weights)
end

"""
    mlrpinv(; kwargs...)
    mlrpinv(X, Y; kwargs...)
    mlrpinv(X, Y, weights::Weight; kwargs...)
    mlrpinv!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Compute a mutiple linear regression model (MLR)  by using 
    a pseudo-inverse. 
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments:
* `noint` : Boolean. Define if the model is computed 
    with an intercept or not.

Safe but can be slower.  

See function `mlr` for examples.
""" 
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
    sqrtD = Diagonal(sqrt.(weights.w))
    if par.noint
        q = nco(Y)
        sqrtDX = sqrtD * X
        tol = sqrt(eps(real(float(one(eltype(sqrtDX))))))      # see ?pinv
        B = pinv(sqrtDX, rtol = tol) * (sqrtD * Y)
        int = zeros(q)'
    else
        xmeans = colmean(X, weights) 
        ymeans = colmean(Y, weights)   
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
        sqrtDX = sqrtD * X
        tol = sqrt(eps(real(float(one(eltype(sqrtDX))))))      # see ?pinv
        B = pinv(sqrtDX, rtol = tol) * (sqrtD * Y)
        int = ymeans' .- xmeans' * B
    end
    Mlr(B, int, weights, par)
end

"""
    mlrpinvn() 
    mlrpinvn(X, Y)
    mlrpinvn(X, Y, weights::Weight)
    mlrpinvn!mlrchol!(X::Matrix, Y::Matrix, 
        weights::Weight)
Compute a mutiple linear regression model (MLR) 
    by using the Normal equations and a pseudo-inverse.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 

Safe and fast for p not too large.

Compute a model with intercept.

See function `mlr` for examples.
""" 
function mlrpinvn(X, Y)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    mlrpinvn(X, Y, weights)
end

function mlrpinvn(X, Y, weights::Weight)
    mlrpinvn!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights)
end

function mlrpinvn!(X::Matrix, Y::Matrix, weights::Weight)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    fcenter!(X, xmeans)
    fcenter!(Y, ymeans)
    XtD = X' * Diagonal(weights.w)
    XtDX = XtD * X
    tol = sqrt(eps(real(float(one(eltype(XtDX))))))
    B = pinv(XtD * X, rtol = tol) * (XtD * Y)
    int = ymeans' .- xmeans' * B
    MlrNoArg(B, int, weights)
end

"""
    mlrvec(; kwargs...)
    mlrvec(X, Y; kwargs...)
    mlrvec(X, Y, weights::Weight; kwargs...)
    mlrvec!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Compute a simple linear regression model (univariate x).
* `x` : Univariate X-data (n).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments:
* `noint` : Boolean. Define if the model is computed 
    with an intercept or not.

See function `mlr` for examples.
""" 
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
    if par.noint
        q = nco(Y)
        xtD = x' * Diagonal(weights.w)
        B = (xtD * Y) ./ (xtD * x)
        int = zeros(q)'
    else
        xmeans = colmean(x, weights) 
        ymeans = colmean(Y, weights)   
        fcenter!(x, xmeans)
        fcenter!(Y, ymeans)
        xtD = x' * Diagonal(weights.w)
        B = (xtD * Y) ./ (xtD * x)
        int = ymeans' .- xmeans' * B
    end
    Mlr(B, int, weights, par)
end

"""
    coef(object::Mlr)
Compute the coefficients of the fitted model.
* `object` : The fitted model.
""" 
function coef(object::Union{Mlr, MlrNoArg, Rrchol})
    (B = object.B, int = object.int)
end

"""
    predict(object::Mlr, X)
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



