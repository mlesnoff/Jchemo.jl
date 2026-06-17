"""
    rrchol(; kwargs...)
    rrchol(X, Y; kwargs...)
    rrchol(X, Y, weights::ProbabilityWeights; kwargs...)
    rrchol!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
Ridge regression (RR) using the Normal equations and a Cholesky factorization.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `lb` : Ridge regularization parameter 'lambda'.
* `scal` : Symbol defining the column scaling of `X`. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

See function `rr` for details and examples.
""" 
rrchol(; kwargs...) = JchemoModel(rrchol, nothing, kwargs)

function rrchol(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    n = nro(X)
    weights = pweight(ones(Q, n))
    rrchol(X, Y, weights; kwargs...)
end

function rrchol(X, Y, weights::ProbabilityWeights; kwargs...)
    rrchol!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function rrchol!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    par = recovkw(ParRr{Q}, kwargs).par
    @assert nco(X) > 1 "The method only works for X with nb columns > 1."
    p = nco(X)
    sqrtw = sqrt.(weights.values)
    lb = Q(par.lb)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)
    xscales = ones(Q, p)
    if par.scal != :none
        colscal = def_colscal(par.scal) 
        xscales .= colscal(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    fcenter!(Y, ymeans)  
    fweightr!(X, sqrtw)
    fweightr!(Y, sqrtw)
    B = cholesky!(Hermitian(X' * X + lb^2 * Diagonal(ones(Q, p)))) \ (X' * Y)
    fweightr!(B, 1 ./ xscales)
    int = ymeans' .- xmeans' * B
    Rrchol(B, int, weights, par)
end


