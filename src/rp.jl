"""
    rp(X, weights = ones(nro(X)); nlv, fun = rpmatli, scal::Bool = false, kwargs ...)
    rp!(X::Matrix, weights = ones(nro(X)); nlv, fun = rpmatli, scal::Bool = false, kwargs ...)
Make a random projection of matrix X.
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. Internally normalized to sum to 1.
* `nlv` : Nb. dimensions on which `X` is projected.
* `fun` : A function of random projection.
* `kwargs` : Optional arguments of function `fun`.
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

## Examples
```julia
X = rand(5, 10)
nlv = 3
fm = rp(X; nlv = nlv)
pnames(fm)
size(fm.P) 
fm.P
fm.T # = X * fm.P 
transf(fm, X[1:2, :])
```
"""
function rp(X; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    rp(X, weights; values(kwargs)...)
end

function rp(X, weights::Weight; kwargs...)
    rp!(copy(ensure_mat(X)), weights; values(kwargs)...)
end

function rp!(X::Matrix, weights::Weight; 
        par = Par())
    @assert in([:gauss, :li])(par.rpmeth) "Wrong value for argument 'rpmeth'."
    Q = eltype(X)
    p = nco(X)
    xmeans = colmean(X, weights)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end
    if par.rpmeth == :gauss
        P = rpmatgauss(p, par.nlv, Q)
    else
        P = rpmatli(p, par.nlv, Q; s = par.s)
    end 
    T = X * P
    Rp(T, P, xmeans, xscales)
end

""" 
    transf(object::Rp, X; nlv = nothing)
Compute "scores" T from a random projection model and a matrix X.
* `object` : The random projection model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. dimensions to consider. If nothing, it is the maximum nb. dimensions.
""" 
function transf(object::Rp, X; nlv = nothing)
    X = ensure_mat(X)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    cscale(X, object.xmeans, object.xscales) * vcol(object.P, 1:nlv)
end

