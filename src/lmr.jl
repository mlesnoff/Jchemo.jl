struct Lmr
    int::Matrix{Float64}
    B::Matrix{Float64}   
    weights::Vector{Float64}
end

"""
    lmrqr(X, Y, weights = ones(size(X, 1)))
    lmrqr!(X, Y, weights = ones(size(X, 1)))
Fit the linear model Y = f(X) (with intercept) by using
the QR algorithm.

- X : matrix (n, p), or vector (n,).
- Y : matrix (n, q), or vector (n,).
- weights: vector (n,).

Safe but little slow.

X and Y are internally centered. 
The inplace version modifies X and Y. 
""" 
lmrqr(X, Y, weights = ones(size(X, 1))) = lmrqr!(copy(X), copy(Y), weights)

function lmrqr!(X, Y, weights = ones(size(X, 1)))
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
    lmrchol!(X, Y, weights = ones(size(X, 1)))
Fit the linear model Y = f(X) (with intercept) by using 
the Normal equations and a Choleski factorization.
- X : matrix (n, p) with p >= 2 (required by function cholesky).
- Y : matrix (n, q), or vector (n,).
- weights: vector (n,).

Faster but can be less accurate (squared element X'X).

X and Y are internally centered. 
The inplace version modifies X and Y. 
""" 
lmrchol(X, Y, weights = ones(size(X, 1))) = lmrchol!(copy(X), copy(Y), weights)

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
    lmrpinv!(X, Y, weights = ones(size(X, 1)))
Fit the linear model Y = f(X) (with intercept) by using 
a pseudo-inverse (B = X^+ * Y). 

- X : matrix (n, p).
- Y : matrix (n, q), or vector (n,).
- weights: vector (n,).

Safe but can be slower. 

X and Y are internally centered. 
The inplace version modifies X and Y (centering). 
""" 
function lmrpinv(X, Y, weights = ones(size(X, 1)))
    res = lmrpinv!(copy(X), copy(Y), weights)
    res
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
    lmrpinvn(X, Y, weights = ones(size(X, 1)))
    lmrpinvn!(X, Y, weights = ones(size(X, 1)))
Fit the linear model Y = f(X) (with intercept) by using 
the Normal equations and a pseudo-inverse.

Safe and fast for p not too large.
- X : matrix (n, p).
- Y : matrix (n, q), or vector (n,).
- weights: vector (n,).

Safe but can be slower. 

X and Y are internally centered. 
The inplace version modifies X and Y. 
""" 
function lmrpinvn(X, Y, weights = ones(size(X, 1)))
    res = lmrpinvn!(copy(X), copy(Y), weights)
    res
end

function lmrpinvn!(X, Y, weights = ones(size(X, 1)))
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
    lmrvec!(x, Y, weights = ones(length(x)))
Fit the linear model Y = f(x) (with intercept)
specifically for univariate x.
- x : matrix (n, p).
- Y : matrix (n, q), or vector (n,).
- weights: vector (n,).

x and Y are internally centered. 
The inplace version modifies x and Y. 
""" 
function lmrvec(x, Y, weights = ones(length(x)))
    res = lmrvec!(copy(x), copy(Y), weights)
    res
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
- object : The fitted model.
""" 
function coef(object::Lmr)
    (int = object.int, B = object.B)
end

"""
    predict(object::Lmr, X)
Compute the Y-predictions from the fitted model.
- object : The fitted model.
- X : Matrix (m, p) for which predictions are computed.
""" 
function predict(object::Lmr, X)
    z = coef(object)
    pred = z.int .+ X * z.B
    (pred = pred,)
end
