struct Lmr
    int::Matrix{Float64}
    B::Matrix{Float64}   
    weights::Vector{Float64}
end

"""
    lmr(X, Y, weights = ones(size(X, 1)))
Compute the linear model `Y` = INT + `X` * B by using
the QR algorithm.
* `X` : matrix (n, p), or vector (n,).
* `Y` : matrix (n, q), or vector (n,).
* `weights` : vector (n,).

Safe but little slow.

`X` and `Y` are internally centered. 

The in-place version modifies `X` and `Y`. 
""" 
function lmr(X, Y, weights = ones(size(X, 1)))
    lmr!(copy(X), copy(Y), weights)
end

function lmr!(X, Y, weights = ones(size(X, 1)))
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = mweights(weights)
    xmeans = colmeans(X, weights) 
    ymeans = colmeans(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    sqrtD = Diagonal(sqrt.(weights))
    B = (sqrtD * X) \ (sqrtD * Y)
    int = ymeans' .- xmeans' * B
    Lmr(int, B, weights)
end

"""
    lmrchol(X, Y, weights = ones(size(X, 1)))
Compute the linear model `Y` = INT + `X` * B by using
the Normal equations and a Choleski factorization.
* `X` : matrix (n, p) with p >= 2 (required by function cholesky).
* `Y` : matrix (n, q), or vector (n,).
* `weights` : vector (n,).

Faster but can be less accurate (squared element X'X).

`X` and `Y` are internally centered. 

The in-place version modifies `X` and `Y`. 
""" 
function lmrchol(X, Y, weights = ones(size(X, 1)))
    lmrchol!(copy(X), copy(Y), weights)
end

function lmrchol!(X, Y, weights = ones(size(X, 1)))
    @assert size(X, 2) > 1 "Method only working for X with > 1 column."
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = mweights(weights)
    xmeans = colmeans(X, weights) 
    ymeans = colmeans(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    XtD = X' * Diagonal(weights)
    B = cholesky!(Hermitian(XtD * X)) \ (XtD * Y)
    int = ymeans' .- xmeans' * B
    Lmr(int, B, weights)
end

"""
    lmrpinv(X, Y, weights = ones(size(X, 1)))
Compute the linear model `Y` = INT + `X` * B by using
a pseudo-inverse (B = X^+ * Y). 

* `X` : matrix (n, p).
* `Y` : matrix (n, q), or vector (n,).
* `weights` : vector (n,).

Safe but can be slower. 

`X` and `Y` are internally centered. 

The in-place version modifies `X` and `Y`. 
""" 
function lmrpinv(X, Y, weights = ones(size(X, 1)))
    lmrpinv!(copy(X), copy(Y), weights)
end

function lmrpinv!(X, Y, weights = ones(size(X, 1)))
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = mweights(weights)
    xmeans = colmeans(X, weights) 
    ymeans = colmeans(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    sqrtD = Diagonal(sqrt.(weights))
    sqrtDX = sqrtD * X
    tol = sqrt(eps(real(float(one(eltype(sqrtDX))))))      # see ?pinv
    B = pinv(sqrtDX, rtol = tol) * (sqrtD * Y)
    int = ymeans' .- xmeans' * B
    Lmr(int, B, weights)
end

"""
    lmrpinv_n(X, Y, weights = ones(size(X, 1)))
Compute the linear model `Y` = INT + `X` * B by using
the Normal equations and a pseudo-inverse.

Safe and fast for p not too large.
* `X` : matrix (n, p).
* `Y` : matrix (n, q), or vector (n,).
* `weights` : vector (n,).

Safe but can be slower. 

`X` and `Y` are internally centered. 

The in-place version modifies `X` and `Y`. 
""" 
function lmrpinv_n(X, Y, weights = ones(size(X, 1)))
    lmrpinv_n!(copy(X), copy(Y), weights)
end

function lmrpinv_n!(X, Y, weights = ones(size(X, 1)))
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = mweights(weights)
    xmeans = colmeans(X, weights) 
    ymeans = colmeans(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    XtD = X' * Diagonal(weights)
    XtDX = XtD * X
    tol = sqrt(eps(real(float(one(eltype(XtDX))))))
    B = pinv(XtD * X, rtol = tol) * (XtD * Y)
    int = ymeans' .- xmeans' * B
    Lmr(int, B, weights)
end

"""
    lmrvec(x, Y, weights = ones(length(x)))
Compute the linear model `Y` = INT + `x` * B   
specifically for univariate x.
* `x` : Matrix (n, 1) or vector (n,).
* `Y` : Matrix (n, q) or vector (n,).
* `weights` : vector (n,).

`x` and `Y` are internally centered. 

The in-place version modifies `x` and `Y`. 
""" 
function lmrvec(x, Y, weights = ones(length(x)))
    lmrvec!(copy(x), copy(Y), weights)
end

function lmrvec!(x, Y, weights = ones(length(x)))
    @assert size(x, 2) == 1 "Method only working for univariate x."
    x = ensure_mat(x)
    Y = ensure_mat(Y)
    weights = mweights(weights)
    xmeans = colmeans(x, weights) 
    ymeans = colmeans(Y, weights)   
    center!(x, xmeans)
    center!(Y, ymeans)
    xtD = x' * Diagonal(weights)
    B = (xtD * Y) ./ (xtD * x)
    int = ymeans' .- xmeans' * B
    Lmr(int, B, weights)
end

"""
    coef(object::Lmr)
Compute the coefficients of the fitted model.
* `object` : The fitted model.
""" 
function coef(object::Lmr)
    (int = object.int, B = object.B)
end

"""
    predict(object::Lmr, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which predictions are computed.
""" 
function predict(object::Lmr, X)
    z = coef(object)
    pred = z.int .+ X * z.B
    (pred = pred,)
end
