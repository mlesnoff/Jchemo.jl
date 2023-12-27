"""
    rrchol(X, Y, weights = ones(nro(X)); lb = .01, 
        scal::Bool = false)
    rrchol!(X::Matrix, Y::Matrix, weights = ones(nro(X)); lb = .01,
        scal::Bool = false)
Ridge regression (RR) using the Normal equations and a Cholesky factorization.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `lb` : Ridge regularization parameter "lambda".
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

`X` and `Y` are internally centered. The model is computed with an intercept. 

See function `rr` for eaxamples.

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
function rrchol(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    rrchol(X, Y, weights; kwargs...)
end

function rrchol(X, Y, weights::Weight; kwargs...)
    rrchol!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; kwargs...)
end

function rrchol!(X::Matrix, Y::Matrix, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert nco(X) > 1 "The method only works for X with nb columns > 1."
    Q = eltype(X)
    p = nco(X)
    lb = convert(Q, par.lb)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    fcenter!(Y, ymeans)  
    XtD = X' * Diagonal(weights.w)
    B = cholesky!(Hermitian(XtD * X + lb^2 * Diagonal(ones(Q, p)))) \ (XtD * Y)
    B .= Diagonal(1 ./ xscales) * B 
    int = ymeans' .- xmeans' * B
    Mlr(B, int, weights, kwargs, par)
end


