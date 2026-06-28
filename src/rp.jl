"""
    rp(; kwargs...)
    rp(X; kwargs...)
    rp(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    rp!(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
Make a random projection of X-data.
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. dimensions on which `X` is projected.
* `meth` : Method of random projection. Possible values are: `:gauss`, `:li`. See the respective functions 
    `rpmatgauss` and `rpmatli` for their keyword arguments.
* `scal` : Symbol defining the column scaling of `X`. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

## Examples
```julia
using Jchemo
n, p = (5, 10)
X = rand(n, p)
nlv = 3
meth = :li ; s = sqrt(p) 
#meth = :gauss
model = rp(; nlv, meth, s)
fit!(model, X)
@names model
@names model.fitm
@head model.fitm.T 
@head model.fitm.V 
transf(model, X[1:2, :])
```
"""
rp(; kwargs...) = JchemoModel(rp, nothing, kwargs)

function rp(X; kwargs...)
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    rp(X, weights; kwargs...)
end

function rp(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    rp!(copy(X), weights; kwargs...)
end

function rp!(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    par = recovkw(ParRp{Q}, kwargs).par 
    @assert in([:gauss, :li])(par.meth) "Wrong value for argument 'meth'."
    p = nco(X)
    ## Centering/scaling of X
    xmeans = colmean(X, weights)
    fcenter!(X, xmeans)
    xscales = ones(Q, p)
    if par.scal != :none
        colscal = def_colscal(par.scal) 
        xscales .= colscal(X, weights)
        fscale!(X, xscales)
    end
    ## End
    if par.meth == :gauss
        V = rpmatgauss(p, par.nlv, Q)
    else
        V = rpmatli(p, par.nlv, Q; s = par.s)
    end 
    T = X * V
    Rp(T, V, xmeans, xscales, par)
end

""" 
    transf(object::Rp, X)
    transf(object::Rp, X, nlv::Int)
Compute scores T from a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which scores T are computed.
* `nlv` : Nb. scores to compute.
""" 
transf(object::Rp, X) = fcscale(X, object.xmeans, object.xscales) * object.V

function transf(object::Rp, X, nlv::Int)
    X = ensure_mat(X)
    a = object.par.nlv
    nlv = isnothing(nlv) ? a : min(nlv, a)
    fcscale(X, object.xmeans, object.xscales) * vcol(object.V, 1:nlv)
end

