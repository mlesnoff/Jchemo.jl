"""
    rp(; kwargs...)
    rp(X; kwargs...)
    rp(X, weights::Weight; kwargs...)
    rp!(X::Matrix, weights::Weight; kwargs...)
Make a random projection of X-data.
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. dimensions on which `X` is projected.
* `mrp` : Method of random projection. Possible
    values are: `:gauss`, `:li`. See the respective 
    functions `rpmatgauss` and `rpmatli` for their 
    keyword arguments.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

## Examples
```julia
n, p = (5, 10)
X = rand(n, p)
nlv = 3
mrp = :li ; s_li = sqrt(p) 
#mrp = :gauss
mod = rp(; nlv, mrp, s_li)
fit!(mod, X)
pnames(mod)
pnames(mod.fm)
@head mod.fm.T 
@head mod.fm.P 
transf(mod, X[1:2, :])
```
"""
function rp(X; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    rp(X, weights; kwargs...)
end

function rp(X, weights::Weight; kwargs...)
    rp!(copy(ensure_mat(X)), weights; 
        kwargs...)
end

function rp!(X::Matrix, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs) 
    @assert in([:gauss, :li])(par.mrp) "Wrong value for argument 'mrp'."
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
    if par.mrp == :gauss
        P = rpmatgauss(p, par.nlv, Q)
    else
        P = rpmatli(p, par.nlv, Q; 
            s_li = par.s_li)
    end 
    T = X * P
    Rp(T, P, xmeans, xscales, kwargs, par)
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

