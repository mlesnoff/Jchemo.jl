"""
    kpca(; kwargs...)
    kpca(X; kwargs...)
    kpca(X, weights::Weight; kwargs...)
Kernel PCA  (Scholkopf et al. 1997, Scholkopf & Smola 2002, Tipping 2001).
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. principal components (PCs) to consider. 
* `kern` : Type of kernel used to compute the Gram matrices.Possible values are: `:krbf`, `:kpol`. See respective functions `krbf` 
    and `kpol` for their keyword arguments.
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

The method is implemented by SVD factorization of the weighted Gram matrix: 
* D^(1/2) * Phi(X) * Phi(X)' * D^(1/2)
where X is the cenetred matrix and D is a diagonal matrix of weights (`weights.w`) of the observations (rows of X).

## References 
Scholkopf, B., Smola, A., MÃ¼ller, K.-R., 1997. Kernel principal component analysis, in: Gerstner, W., Germond, A., Hasler, 
M., Nicoud, J.-D. (Eds.), Artificial Neural Networks, ICANN 97, Lecture Notes in Computer Science. Springer, Berlin, 
Heidelberg, pp. 583-588. https://doi.org/10.1007/BFb0020217

Scholkopf, B., Smola, A.J., 2002. Learning with kernels: support vector machines, regularization, optimization, and beyond, Adaptive 
computation and machine learning. MIT Press, Cambridge, Mass.

Tipping, M.E., 2001. Sparse kernel principal component analysis. Advances in neural information processing systems, MIT Press. 
http://papers.nips.cc/paper/1791-sparse-kernel-principal-component-analysis.pdf

## Examples
```julia
using Jchemo, JchemoData, JLD2 
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
@names dat
@head dat.X
X = dat.X[:, 1:4]
n = nro(X)
ntest = 30
s = samprand(n, ntest) 
Xtrain = X[s.train, :]
Xtest = X[s.test, :]

nlv = 3
kern = :krbf ; gamma = 1e-4
model = kpca(; nlv, kern, gamma) 
fit!(model, Xtrain)
fitm = model.fitm ;
@names fitm

@head transf(model, Xtrain)
@head fitm.T

@head transf(model, Xtest) 

res = summary(model) ;
@names res
res.explvarx
```
""" 
kpca(; kwargs...) = JchemoModel(kpca, nothing, kwargs)

function kpca(X; kwargs...)
    X = ensure_mat(X)
    Q = eltype(X)
    weights = mweight(ones(Q, nro(X)))
    kpca(X, weights; kwargs...)
end

function kpca(X, weights::Weight; kwargs...)
    par = recovkw(ParKpca, kwargs).par
    @assert in([:krbf ; :kpol])(par.kern) "Wrong value for argument 'kern'." 
    X = ensure_mat(X)
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        X = fscale(X, xscales)
    end
    fkern = eval(Meta.parse(string("Jchemo.", par.kern)))  
    K = fkern(X, X; kwargs...)  # in the future?: fkern!(K, X, X; kwargs...)
    sqrtw = sqrt.(weights.w)
    Kt = K'    
    DKt = fweight(Kt, weights.w)
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- vtot .+ sum(fweight(DKt', weights.w))    # = K .- vtot' .- vtot .+ sum(D * DKt')
    Kd = fweight(Kc, sqrtw) * Diagonal(sqrtw)    # = sqrtD * Kc * sqrtD
    res = LinearAlgebra.svd(Kd)
    U = res.V[:, 1:nlv]
    eig = res.S
    eig[eig .< 0] .= 0
    sv = sqrt.(eig)
    V = fweight(fscale(U, sv[1:nlv]), sqrtw)   # In the future: fscale!
    T = Kc * V       # T = Kc * V = D^(-1/2) * U * Delta
    Kpca(X, Kt, T, V, sv, eig, DKt, vtot, xscales, weights, kwargs, par)
end

""" 
    transf(object::Kpca, X; nlv = nothing)
Compute PCs (scores T) from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which PCs are computed.
* `nlv` : Nb. PCs to compute.
""" 
function transf(object::Kpca, X; nlv = nothing)
    a = object.par.nlv
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(fscale(X, object.xscales), object.X; object.kwargs...)
    w = object.weights.w
    DKt = fweight(K', w)
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- object.vtot .+ sum(fweight(object.DKt', w))
    T = Kc * @view(object.V[:, 1:nlv])
    T
end

"""
    summary(object::Kpca)
Summarize the fitted model.
* `object` : The fitted model.
""" 
function Base.summary(object::Kpca)
    nlv = nco(object.T)
    tt = colnorm(object.T, object.weights).^2
    sstot = sum(object.eig)
    pvar = tt / sstot
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(lv = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    (explvarx = explvarx,)
end


