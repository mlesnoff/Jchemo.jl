"""
    mlr(X, Y, weights = ones(nro(X)); noint::Bool = false)
    mlr!(X::Matrix, Y::Matrix, weights = ones(nro(X)); noint::Bool = false)
Compute a mutiple linear regression model (MLR) by using the QR algorithm.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Internally normalized to sum to 1.
* `noint` : Define if the model is computed with an intercept or not.

Safe but can be little slower than other methods.

## Examples
```julia
using JchemoData, JLD2, CairoMakie, StatsBase
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
pnames(dat)
summ(dat.X)

X = Matrix(dat.X[:, 2:4]) 
y = dat.X[:, 1]
n = nro(X)
ntrain = 120
s = sample(1:n, ntrain; replace = false) 
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

fm = mlr(Xtrain, ytrain) ;
#fm = mlrchol(Xtrain, ytrain) ;
#fm = mlrpinv(Xtrain, ytrain) ;
#fm = mlrpinvn(Xtrain, ytrain) ;
pnames(fm)
res = Jchemo.predict(fm, Xtest)
rmsep(res.pred, ytest)
plotxy(pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f    

zcoef = Jchemo.coef(fm) 
zcoef.int 
zcoef.B 

fm = mlr(Xtrain, ytrain; noint = true) ;
zcoef = Jchemo.coef(fm) 
zcoef.int 
zcoef.B

fm = mlr(Xtrain[:, 1], ytrain) ;
#fm = mlrvec(Xtrain[:, 1], ytrain) ;
zcoef = Jchemo.coef(fm) 
zcoef.int 
zcoef.B
```
""" 
function mlr(X, Y, weights = ones(nro(X)); 
        noint::Bool = false)
    mlr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights;
        noint = noint)
end

function mlr!(X::Matrix, Y::Matrix, weights = ones(nro(X)); 
        noint::Bool = false)
    weights = mweight(weights)
    sqrtD = Diagonal(sqrt.(weights))
    if noint
        q = nco(Y)
        B = (sqrtD * X) \ (sqrtD * Y)
        int = zeros(q)'
    else
        xmeans = colmean(X, weights) 
        ymeans = colmean(Y, weights)   
        center!(X, xmeans)
        center!(Y, ymeans)
        B = (sqrtD * X) \ (sqrtD * Y)
        int = ymeans' .- xmeans' * B
    end
    Mlr(B, int, weights)
end

"""
    mlrchol(X, Y, weights = ones(nro(X)))
    mlrchol!(X::Matrix, Y::Matrix, weights = ones(nro(X)))
Compute a mutiple linear regression model (MLR) 
using the Normal equations and a Choleski factorization.
* `X` : X-data, with nb. columns >= 2 (required by function cholesky).
* `Y` : Y-data.
* `weights` : Weights of the observations. Internally normalized to sum to 1. 

Compute a model with intercept.

Faster but can be less accurate (squared element X'X).

See `?mlr` for examples.
""" 
function mlrchol(X, Y, weights = ones(nro(X)))
    mlrchol!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights)
end

function mlrchol!(X::Matrix, Y::Matrix, weights = ones(nro(X)))
    @assert nco(X) > 1 "Method only working for X with > 1 column."
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    XtD = X' * Diagonal(weights.w)
    B = cholesky!(Hermitian(XtD * X)) \ (XtD * Y)
    int = ymeans' .- xmeans' * B
    Mlr(B, int, weights)
end

"""
    mlrpinv(X, Y, weights = ones(nro(X)); noint::Bool = false)
    mlrpinv!(X::Matrix, Y::Matrix, weights = ones(nro(X)); noint::Bool = false)
Compute a mutiple linear regression model (MLR)  by using a pseudo-inverse. 
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `noint` : Define if the model is computed with an intercept or not.

Safe but can be slower.  

See `?mlr` for examples.
""" 
function mlrpinv(X, Y, weights = ones(nro(X)); 
        noint::Bool = false)
    mlrpinv!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; 
        noint = noint)
end

function mlrpinv!(X::Matrix, Y::Matrix, weights = ones(nro(X)); 
        noint::Bool = false)
    weights = mweight(weights)
    sqrtD = Diagonal(sqrt.(weights))
    if noint
        q = nco(Y)
        sqrtDX = sqrtD * X
        tol = sqrt(eps(real(float(one(eltype(sqrtDX))))))      # see ?pinv
        B = pinv(sqrtDX, rtol = tol) * (sqrtD * Y)
        int = zeros(q)'
    else
        xmeans = colmean(X, weights) 
        ymeans = colmean(Y, weights)   
        center!(X, xmeans)
        center!(Y, ymeans)
        sqrtDX = sqrtD * X
        tol = sqrt(eps(real(float(one(eltype(sqrtDX))))))      # see ?pinv
        B = pinv(sqrtDX, rtol = tol) * (sqrtD * Y)
        int = ymeans' .- xmeans' * B
    end
    Mlr(B, int, weights)
end

"""
    mlrpinvn(X, Y, weights = ones(nro(X)))
    mlrpinvn!(X::Matrix, Y::Matrix, weights = ones(nro(X)))
Compute a mutiple linear regression model (MLR) 
by using the Normal equations and a pseudo-inverse.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations. Internally normalized to sum to 1. 

Safe and fast for p not too large.

Compute a model with intercept.

See `?mlr` for examples.
""" 
function mlrpinvn(X, Y, weights = ones(nro(X)))
    mlrpinvn!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights)
end

function mlrpinvn!(X::Matrix, Y::Matrix, weights = ones(nro(X)))
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    XtD = X' * Diagonal(weights.w)
    XtDX = XtD * X
    tol = sqrt(eps(real(float(one(eltype(XtDX))))))
    B = pinv(XtD * X, rtol = tol) * (XtD * Y)
    int = ymeans' .- xmeans' * B
    Mlr(B, int, weights)
end

"""
    mlrvec(x, Y, weights = ones(length(x));
        noint::Bool = false)
    mlrvec!(x::Matrix, Y::Matrix, weights = ones(length(x));
        noint::Bool = false)
Compute a simple linear regression model (univariate x).
* `x` : Univariate X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations. Internally normalized to sum to 1. 

Compute a model with intercept.

See `?mlr` for examples.
""" 
     
function mlrvec(x, Y, weights = ones(length(x));
        noint::Bool = false)
    mlrvec!(copy(ensure_mat(x)), copy(ensure_mat(Y)), weights;
        noint = noint)
end

function mlrvec!(x::Matrix, Y::Matrix, weights = ones(length(x));
        noint::Bool = false)
    @assert nco(x) == 1 "Method only working for univariate x."
    weights = mweight(weights)
    if noint
        q = nco(Y)
        xtD = x' * Diagonal(weights.w)
        B = (xtD * Y) ./ (xtD * x)
        int = zeros(q)'
    else
        xmeans = colmean(x, weights) 
        ymeans = colmean(Y, weights)   
        center!(x, xmeans)
        center!(Y, ymeans)
        xtD = x' * Diagonal(weights.w)
        B = (xtD * Y) ./ (xtD * x)
        int = ymeans' .- xmeans' * B
    end
    Mlr(B, int, weights)
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



