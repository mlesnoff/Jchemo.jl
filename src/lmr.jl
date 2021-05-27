struct Lmr2
    int::Matrix{Float64}
    B::Matrix{Float64}   
    weights::Vector{Float64}
end

"""
lmrqr!(X, Y, weights = ones(size(X, 1)))
Usual QR solving of X * B = Y
Safe but little slow.
- X {Float64}: matrix (n, p) with p >= 1, or vector (n,)
- Y {Float64}: matrix (n, q) with q >= 1, or vector (n,)
- weights: vector (n,)
X and Y are internally centered.
For saving allocation memory, the centering is done "inplace",
which modifies externally X and Y. 
""" 
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
    Lmr2(int, B, weights)
end

function lmrqr(X, Y, weights = ones(size(X, 1)))
    res = lmrqr!(copy(X), copy(Y), weights)
    res
end

"""
lmrchol!(X, Y, weights = ones(size(X, 1)))
Uses Normal equations and Choleski factorization.
Faster but can be less accurate (squared element X'X).
X must have > 1 column (required by function cholesky).
""" 
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
    B = cholesky(Hermitian(XtD * X)) \ (XtD * Y)
    int = ymeans' .- xmeans' * B
    Lmr2(int, B, weights)
end

function lmrchol(X, Y, weights = ones(size(X, 1)))
    res = lmrchol!(copy(X), copy(Y), weights)
    res
end

"""
lmrpinv!(X)
Pseudo-inverse B = X+ * Y
Safe 
""" 
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
    Lmr2(int, B, weights)
end

function lmrpinv(X, Y, weights = ones(size(X, 1)))
    res = lmrpinv!(copy(X), copy(Y), weights)
    res
end

"""
lmrpinv2!(X)
Uses Normal equations and pseudo-inverse 
Safe and fast for p not too large
""" 
function lmrpinv2!(X, Y, weights = ones(size(X, 1)))
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
    Lmr2(int, B, weights)
end

function lmrpinv2(X, Y, weights = ones(size(X, 1)))
    res = lmrpinv2!(copy(X), copy(Y), weights)
    res
end


"""
lmrvec(X)
Specific (faster) for univariate X
""" 
function lmrvec!(X, Y, weights = ones(size(X, 1)))
    @assert size(X, 2) == 1 "Method only working for univariate X."
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = mweights(weights)
    xmeans = colmeans(X, weights) 
    ymeans = colmeans(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    XtD = X' * Diagonal(weights)
    B = (XtD * Y) ./ (XtD * X)
    int = ymeans' .- xmeans' * B
    Lmr2(int, B, weights)
end

function lmrvec(X, Y, weights = ones(size(X, 1)))
    res = lmrvec!(copy(X), copy(Y), weights)
    res
end