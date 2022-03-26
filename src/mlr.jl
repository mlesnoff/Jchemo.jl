struct Mlr
    B::Matrix{Float64}   
    int::Matrix{Float64}
    weights::Vector{Float64}
end

"""
    mlr(X, Y, weights = ones(size(X, 1)); noint = false)
Compute a mutiple linear regression model (MLR) by using the QR algorithm.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `noint` : Define if the model is computed with an intercept or not.

Safe but little slower.
""" 
function mlr(X, Y, weights = ones(size(X, 1)); noint = false)
    mlr!(copy(X), copy(Y), weights; noint = noint)
end

function mlr!(X, Y, weights = ones(size(X, 1)); noint = false)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
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
    mlrchol(X, Y, weights = ones(size(X, 1)); noint = false)
Compute a mutiple linear regression model (MLR) 
using the Normal equations and a Choleski factorization.
* `X` : X-data, with nb. columns >= 2 (required by function cholesky).
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `noint` : Define if the model is computed with an intercept or not.

Faster but can be less accurate (squared element X'X).
""" 
function mlrchol(X, Y, weights = ones(size(X, 1)))
    mlrchol!(copy(X), copy(Y), weights)
end

function mlrchol!(X, Y, weights = ones(size(X, 1)))
    @assert size(X, 2) > 1 "Method only working for X with > 1 column."
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    XtD = X' * Diagonal(weights)
    B = cholesky!(Hermitian(XtD * X)) \ (XtD * Y)
    int = ymeans' .- xmeans' * B
    Mlr(B, int, weights)
end

"""
    mlrpinv(X, Y, weights = ones(size(X, 1)); noint = false)
Compute a mutiple linear regression model (MLR)  by using a pseudo-inverse. 
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `noint` : Define if the model is computed with an intercept or not.

Safe but can be slower.  
""" 
function mlrpinv(X, Y, weights = ones(size(X, 1)); noint = false)
    mlrpinv!(copy(X), copy(Y), weights; noint = noint)
end

function mlrpinv!(X, Y, weights = ones(size(X, 1)); noint = false)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
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
    mlrpinv_n(X, Y, weights = ones(size(X, 1)); noint = false)
Compute a mutiple linear regression model (MLR) 
by using the Normal equations and a pseudo-inverse.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `noint` : Define if the model is computed with an intercept or not.

Safe and fast for p not too large.
""" 
function mlrpinv_n(X, Y, weights = ones(size(X, 1)))
    mlrpinv_n!(copy(X), copy(Y), weights)
end

function mlrpinv_n!(X, Y, weights = ones(size(X, 1)))
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    XtD = X' * Diagonal(weights)
    XtDX = XtD * X
    tol = sqrt(eps(real(float(one(eltype(XtDX))))))
    B = pinv(XtD * X, rtol = tol) * (XtD * Y)
    int = ymeans' .- xmeans' * B
    Mlr(B, int, weights)
end

"""
    mlrvec(x, Y, weights = ones(length(x)); noint = false)
Compute a simple linear regression model (univariate x).
* `x` : Univariate X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `noint` : Define if the model is computed with an intercept or not.
""" 
function mlrvec(x, Y, weights = ones(length(x)))
    mlrvec!(copy(x), copy(Y), weights)
end

function mlrvec!(x, Y, weights = ones(length(x)))
    @assert size(x, 2) == 1 "Method only working for univariate x."
    x = ensure_mat(x)
    Y = ensure_mat(Y)
    weights = mweight(weights)
    xmeans = colmean(x, weights) 
    ymeans = colmean(Y, weights)   
    center!(x, xmeans)
    center!(Y, ymeans)
    xtD = x' * Diagonal(weights)
    B = (xtD * Y) ./ (xtD * x)
    int = ymeans' .- xmeans' * B
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



