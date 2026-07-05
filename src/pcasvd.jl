"""
    pcasvd(; kwargs...)
    pcasvd(X; kwargs...)
    pcasvd(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    pcasvd!(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
PCA by SVD factorization.
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. of principal components (PCs).
* `scal` : Symbol defining the column scaling of `X`. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

Let us note D the (n, n) diagonal matrix of weights (`weights.values`) and X the centered matrix in metric D.
The function minimizes ||X - T * V'||^2  in metric D, by computing a SVD factorization of sqrt(D) * X:
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
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    pcasvd(X, weights; kwargs...)
end

function pcasvd(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    pcasvd!(copy(X), weights; kwargs...)
end

function pcasvd!(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    par = recovkw(ParPca, kwargs).par
    n, p = size(X)
    nlv = min(n, p, par.nlv)
    par.nlv = nlv
    ## Centering/scaling X
    xmeans = colmean(X, weights)
    fcenter!(X, xmeans)
    xscales = ones(Q, p)
    if par.scal != :none
        colscal = def_colscal(par.scal) 
        xscales .= colscal(X, weights)
        fscale!(X, xscales)
    end
    ## by default in LinearAlgebra.svd, "full = false" ==> [1:min(n, p)]
    sqrtw = sqrt.(weights.values)
    fweightr!(X, sqrtw)
    res = LinearAlgebra.svd!(X)
    V = res.V[:, 1:nlv]
    sv = res.S   
    sv[sv .< 0] .= 0
    T = vcol(res.U, 1:nlv) * Diagonal(sv[1:nlv])
    fweightr!(T, 1 ./ sqrtw)
    Pca(T, V, sv, xmeans, xscales, weights, par) 
end

""" 
    transf(object::Union{Pca, Pcanipals, Fda}, X)
    transf(object::Union{Pca, Pcanipals, Fda}, X, nlv::Int)
Compute principal components (PCs = scores T) from a fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which PCs are computed.
* `nlv` : Nb. PCs to compute.
""" 
transf(object::Union{Pca, Pcanipals, Fda}, X) = fcscale(X, object.xmeans, object.xscales) * object.V

function transf(object::Union{Pca, Pcanipals, Fda}, X, nlv::Int)
    nlv = min(nlv, object.par.nlv)
    fcscale(X, object.xmeans, object.xscales) * vcol(object.V, 1:nlv)
end

"""
    summary(object::Union{Pca, Pcanipals}, X)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Union{Pca, Pcanipals}, X)
    X = ensure_mat(X)
    nlv = nco(object.T)
    weights = object.weights
    X = fcscale(X, object.xmeans, object.xscales)
    sstot = frob2(X, weights)  # = (||X||_D)^2 = tr(X' * D * X)
    TT = fweightr(object.T.^2, weights.values)  # matrix required for 'contr_ind'
    tt = colsum(TT) 
    ## = colnorm(object.T, weights).^2 
    ## = diag(T' * D * T) 
    ## = object.sv[1:nlv].^2
    pvar = tt / sstot
    cumpvar = cumsum(pvar)
    zrd = vec(rd(X, object.T, weights))
    explvarx = DataFrame(nlv = collect(1:nlv), rd = zrd, var = tt, pvar = pvar, cumpvar = cumpvar)
    nam = string.("lv", 1:nlv)
    contr_ind = DataFrame(fscale(TT, tt), nam)
    contr_var = DataFrame(object.V.^2, nam)
    C = X' * fweightr(fscale(object.T, sqrt.(tt)), weights.values)  # V_tild = X' * D * T_normed
    coord_var = DataFrame(C, nam)
    cor_circle = DataFrame(corm(X, object.T, weights), nam)
    (explvarx = explvarx, contr_ind, contr_var, coord_var, cor_circle)
end


