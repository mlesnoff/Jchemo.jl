struct RrSvd
    V::Array{Float64}
    TtDY::Array{Float64}
    sv::Vector{Float64}
    lb::Float64
    xmeans::Vector{Float64}
    ymeans::Vector{Float64}
    weights::Vector{Float64}
end

"""
    rrsvd(X, Y, weights = ones(size(X, 1)); lb = .01)
Ridge regression (RR) implemented by SVD factorization.
* `X` : matrix (n, p), or vector (n,).
* `Y` : matrix (n, q), or vector (n,).
* `weights` : vector (n,).
* `lb` : A value of the regularization parameter "lambda".

`X` and `Y` are internally centered. 

The in-place version modifies `X` and `Y`. 

## References 

Cule, E., De Iorio, M., 2012. A semi-automatic method to guide the choice 
of ridge parameter in ridge regression. arXiv:1205.0686.

Hastie, T., Tibshirani, R., 2004. Efficient quadratic regularization 
for expression arrays. Biostatistics 5, 329-340. https://doi.org/10.1093/biostatistics/kxh010

Hastie, T., Tibshirani, R., Friedman, J., 2009. The elements of statistical learning: data mining, 
inference, and prediction, 2nd ed. Springer, New York.

Hoerl, A.E., Kennard, R.W., 1970. Ridge Regression: Biased Estimation for Nonorthogonal Problems. 
Technometrics 12, 55-67. https://doi.org/10.1080/00401706.1970.10488634
""" 
function rrsvd(X, Y, weights = ones(size(X, 1)); lb = .01)
    rrsvd!(copy(X), copy(Y), weights; lb = lb)
end

function rrsvd!(X, Y, weights = ones(size(X, 1)); lb = .01)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = mweights(weights)
    sqrtw = sqrt.(weights)
    xmeans = colmeans(X, weights) 
    ymeans = colmeans(Y, weights)   
    center!(X, xmeans)
    sqrtD = Diagonal(sqrtw)
    res = LinearAlgebra.svd!(sqrtD * X)
    sv = res.S
    TtDY = Diagonal(sv) * res.U' * (sqrtD * Y)
    RrSvd(res.V, TtDY, sv, lb, xmeans, ymeans, weights)
end

"""
    rrchol(X, Y, weights = ones(size(X, 1)); lb = .01)
Ridge regression (RR) using the Normal equations and implemented by Cholesky factorization.
* `X` : matrix (n, p) with p >= 2  (required by function cholesky).
* `Y` : matrix (n, q), or vector (n,).
* `weights` : vector (n,).
* `lb` : A value of the regularization parameter "lambda".

`X` and `Y` are internally centered. 

The in-place version modifies `X` and `Y`. 

## References 

Cule, E., De Iorio, M., 2012. A semi-automatic method to guide the choice 
of ridge parameter in ridge regression. arXiv:1205.0686.

Hastie, T., Tibshirani, R., 2004. Efficient quadratic regularization 
for expression arrays. Biostatistics 5, 329-340. https://doi.org/10.1093/biostatistics/kxh010

Hastie, T., Tibshirani, R., Friedman, J., 2009. The elements of statistical learning: data mining, 
inference, and prediction, 2nd ed. Springer, New York.

Hoerl, A.E., Kennard, R.W., 1970. Ridge Regression: Biased Estimation for Nonorthogonal Problems. 
Technometrics 12, 55-67. https://doi.org/10.1080/00401706.1970.10488634
""" 
function rrchol(X, Y, weights = ones(size(X, 1)); lb = .01)
    rrchol!(copy(X), copy(Y), weights; lb = lb)
end

function rrchol!(X, Y, weights = ones(size(X, 1)); lb = .01)
    @assert size(X, 2) > 1 "Method only working for X with > 1 column."
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    p = size(X, 2)
    weights = mweights(weights)
    xmeans = colmeans(X, weights) 
    ymeans = colmeans(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    XtD = X' * Diagonal(weights)
    B = cholesky!(Hermitian(XtD * X + lb^2 * Diagonal(ones(p)))) \ (XtD * Y)
    int = ymeans' .- xmeans' * B
    Lmr(int, B, weights)
end

"""
    coef(object::RrSvd; lb = nothing)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `lb` : A value of the regularization parameter "lambda".
If nothing, it is the parameter stored in the fitted model.
""" 
function coef(object::RrSvd; lb = nothing)
    isnothing(lb) ? lb = object.lb : nothing
    eig = object.sv.^2
    z = 1 ./ (eig .+ lb^2)
    beta = Diagonal(z) * object.TtDY
    B = object.V * beta
    int = object.ymeans' .- object.xmeans' * B
    tr = sum(eig .* z)
    (int = int, B = B, df = 1 + tr)
end

"""
    predict(object::RrSvd, X; lb = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which predictions are computed.
* `lb` : Regularization parameter, or collection of regularization parameters, "lambda" to consider. 
If nothing, it is the parameter stored in the fitted model.
""" 
function predict(object::RrSvd, X; lb = nothing)
    isnothing(lb) ? lb = object.lb : nothing
    le_lb = length(lb)
    pred = list(le_lb)
    @inbounds for i = 1:le_lb
        z = coef(object; lb = lb[i])
        pred[i] = z.int .+ X * z.B
    end 
    le_lb == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end
