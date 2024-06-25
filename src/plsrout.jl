"""
    plsrout(X, Y; kwargs...)
    plsrout(X, Y, weights::Weight; kwargs...)
    pcaout!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Robust PLSR using outlierness.
* `X` : X-data (n, p). 
* `Y` : Y-data (n, q). 
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. of latent variables (LVs).
* `prm` : Proportion of the data removed (hard rejection of outliers) 
    for each outlierness measure.
* `scal` : Boolean. If `true`, `X`-data are scaled in the 
    functions computing the outlierness and the weighted PLSR. 

Robust PLSR combining outlyingness measures and weighted PLSR (WPLSR).
This is the same principle as function `pcaout` (see the help page) but
the final step is a weighted PLSR instead of a weighted PCA.  

## Examples
```julia
```
""" 
function plsrout(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plsrout(X, Y, weights; kwargs...)
end

function plsrout(X, Y, weights::Weight; kwargs...)
    plsrout!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function plsrout!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs) 
    n, p = size(X)
    nlvout = 30
    P = rand(0:1, p, nlvout)
    d = similar(X, n)
    d .= outstah(X, P; scal = par.scal).d
    w = talworth(d; a = quantile(d, 1 - par.prm))
    d .= outeucl(X; scal = par.scal).d
    w .*= talworth(d; a = quantile(d, 1 - par.prm))
    w .*= weights.w
    w[isequal.(w, 0)] .= 1e-10
    plskern(X, Y, mweight(w); kwargs...)
end
