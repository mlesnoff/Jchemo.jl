"""
    rrchol(; kwargs...)
    rrchol(X, Y; kwargs...)
    rrchol(X, Y, weights::Weight; kwargs...)
    rrchol!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Ridge regression (RR) using the Normal equations and a Cholesky factorization.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g., function `mweight`).
Keyword arguments:
* `lb` : Ridge regularization parameter 'lambda'.
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

See function `rr` for details and examples.
""" 
rrchol(; kwargs...) = JchemoModel(rrchol, nothing, kwargs)

function rrchol(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    rrchol(X, Y, weights; kwargs...)
end

function rrchol(X, Y, weights::Weight; kwargs...)
    rrchol!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function rrchol!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParRr, kwargs).par
    @assert nco(X) > 1 "The method only works for X with nb columns > 1."
    Q = eltype(X)
    p = nco(X)
    sqrtw = sqrt.(weights.w)
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
    rweight!(X, sqrtw)
    rweight!(Y, sqrtw)
    B = cholesky!(Hermitian(X' * X + lb^2 * Diagonal(ones(Q, p)))) \ (X' * Y)
    rweight!(B, 1 ./ xscales)
    int = ymeans' .- xmeans' * B
    Rrchol(B, int, weights, par)
end


