"""
    pcasvd(; kwargs...)
    pcasvd(X; kwargs...)
    pcasvd(X, weights::Weight; kwargs...)
    pcasvd!(X::Matrix, weights::Weight; kwargs...)
PCA by SVD factorization.
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. of principal components (PCs).
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

Let us note D the (n, n) diagonal matrix of weights
(`weights.w`) and X the centered matrix in metric D.
The function minimizes ||X - T * V'||^2  in metric D, by 
computing a SVD factorization of sqrt(D) * X:

* sqrt(D) * X ~ U * S * V'

Outputs are:
* `T` = D^(-1/2) * U * S
* `V` = V
* The diagonal of S   

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie 
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
@names dat
@head dat.X
X = dat.X[:, 1:4]
n = nro(X)
ntest = 30
s = samprand(n, ntest) 
@head Xtrain = X[s.train, :]
@head Xtest = X[s.test, :]

nlv = 3
model = pcasvd(; nlv)
#model = pcaeigen(; nlv)
#model = pcaeigenk(; nlv)
#model = pcanipals(; nlv)
fit!(model, Xtrain)
@names model
@names model.fitm
@head T = model.fitm.T
## Same as:
@head transf(model, X)
T' * T
@head V = model.fitm.V
V' * V

@head Ttest = transf(model, Xtest)

res = summary(model, Xtrain) ;
@names res
res.explvarx
res.contr_var
res.coord_var
res.cor_circle
```
""" 
pcasvd(; kwargs...) = JchemoModel(pcasvd, nothing, kwargs)

function pcasvd(X; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    pcasvd(X, weights; kwargs...)
end

function pcasvd(X, weights::Weight; kwargs...)
    pcasvd!(copy(ensure_mat(X)), weights; kwargs...)
end

function pcasvd!(X::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParPca, kwargs).par
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    xmeans = colmean(X, weights)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    ## by default in LinearAlgebra.svd, "full = false" ==> [1:min(n, p)]
    sqrtw = sqrt.(weights.w)
    fweight!(X, sqrtw)
    res = LinearAlgebra.svd!(X)
    V = res.V[:, 1:nlv]
    sv = res.S   
    sv[sv .< 0] .= 0
    T = vcol(res.U, 1:nlv) * Diagonal(sv[1:nlv])
    fweight!(T, 1 ./ sqrtw)
    Pca(T, V, sv, xmeans, xscales, weights, nothing, par) 
end

""" 
    transf(object::Union{Pca, Fda}, X; nlv = nothing)
Compute principal components (PCs = scores T) from a fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which PCs are computed.
* `nlv` : Nb. PCs to compute.
""" 
function transf(object::Union{Pca, Fda}, X; nlv = nothing)
    X = ensure_mat(X)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    fcscale(X, object.xmeans, object.xscales) * vcol(object.V, 1:nlv)
end

"""
    summary(object::Pca, X)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Pca, X)
    X = ensure_mat(X)
    nlv = nco(object.T)
    weights = object.weights
    X = fcscale(X, object.xmeans, object.xscales)
    sstot = frob2(X, weights)  # = (||X||_D)^2 = tr(X' * D * X)
    TT = fweight(object.T.^2, weights.w)  # matrix required for 'contr_ind'
    tt = colsum(TT) 
    ## = colnorm(object.T, weights).^2 
    ## = diag(T' * D * T) 
    ## = object.sv[1:nlv].^2
    pvar = tt / sstot
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(nlv = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    nam = string.("lv", 1:nlv)
    contr_ind = DataFrame(fscale(TT, tt), nam)
    contr_var = DataFrame(object.V.^2, nam)
    C = X' * fweight(fscale(object.T, sqrt.(tt)), weights.w)  # V_tild = X' * D * T_normed
    coord_var = DataFrame(C, nam)
    cor_circle = DataFrame(corm(X, object.T, weights), nam)
    (explvarx = explvarx, contr_ind, contr_var, coord_var, cor_circle)
end


