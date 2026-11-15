"""
    rrchol(; kwargs...)
    rrchol(X, Y; kwargs...)
    rrchol(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
    rrchol!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
Ridge regression (RR) using the Normal equations and a Cholesky factorization.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `lb` : Ridge regularization parameter 'lambda'.
* `scal` : Symbol defining the column scaling of `X`. Possible values are: `:none` (default), 
    `:std` (uncorrected STD), `:prt` (Pareto), and `:mad` (MAD).

See function `rr` for details and examples.
""" 
rrchol(; kwargs...) = JchemoModel(rrchol, nothing, kwargs)

function rrchol(X, Y; kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = pweight(ones(eltype(X), nro(X)))
    rrchol(X, Y, weights; kwargs...)
end

function rrchol(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
    rrchol!(copy(X), copy(Y), weights; kwargs...)
end

function rrchol!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
    par = recovkw(ParRr{Q}, kwargs).par
    @assert nco(X) > 1 "The method only works for X with nb columns > 1."
    p = nco(X)
    sqrtw = sqrt.(weights.values)
    # Centering and scaling
    # No need to scale Y
    xcenters = colmean(X, weights) 
    ycenters = colmean(Y, weights)
    fcenter!(X, xcenters)
    fcenter!(Y, ycenters)
    xscales = colscal(X, weights, par.scal)
    if par.scal != :none
        fscale!(X, xscales)
    end
    # End
    fweightr!(X, sqrtw)
    fweightr!(Y, sqrtw)
    B = cholesky!(Hermitian(X' * X + par.lb^2 * Diagonal(ones(Q, p)))) \ (X' * Y)
    fweightr!(B, 1 ./ xscales)
    int = ycenters' .- xcenters' * B
    Rrchol(B, int, weights, par)
end


