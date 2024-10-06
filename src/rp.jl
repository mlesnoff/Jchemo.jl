"""
    rp(X; kwargs...)
    rp(X, weights::Weight; kwargs...)
    rp!(X::Matrix, weights::Weight; kwargs...)
Make a random projection of X-data.
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. dimensions on which `X` is projected.
* `meth` : Method of random projection. Possible
    values are: `:gauss`, `:li`. See the respective 
    functions `rpmatgauss` and `rpmatli` for their 
    keyword arguments.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

## Examples
```julia
using Jchemo
n, p = (5, 10)
X = rand(n, p)
nlv = 3
meth = :li ; s = sqrt(p) 
#meth = :gauss
model = mod_(rp; nlv, meth, s)
fit!(model, X)
pnames(model)
pnames(model.fm)
@head model.fm.T 
@head model.fm.P 
transf(model, X[1:2, :])
```
"""
function rp(X; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    rp(X, weights; kwargs...)
end

function rp(X, weights::Weight; kwargs...)
    rp!(copy(ensure_mat(X)), weights; kwargs...)
end

function rp!(X::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParRp, kwargs).par 
    @assert in([:gauss, :li])(par.meth) "Wrong value for argument 'meth'."
    Q = eltype(X)
    p = nco(X)
    xmeans = colmean(X, weights)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    if par.meth == :gauss
        P = rpmatgauss(p, par.nlv, Q)
    else
        P = rpmatli(p, par.nlv, Q; s = par.s)
    end 
    T = X * P
    Rp(T, P, xmeans, xscales, par)
end

""" 
    transf(object::Rp, X; nlv = nothing)
Compute scores T from a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which scores T are computed.
* `nlv` : Nb. scores to compute.
""" 
function transf(object::Rp, X; nlv = nothing)
    X = ensure_mat(X)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    fcscale(X, object.xmeans, object.xscales) * vcol(object.P, 1:nlv)
end

