"""
    kpca(; kwargs...)
    kpca(X; kwargs...)
    kpca(X, weights::Weight; 
        kwargs...)
Kernel PCA  (Scholkopf et al. 1997, Scholkopf & Smola 2002, 
    Tipping 2001).
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. principal components (PCs) to consider. 
* `kern` : Type of kernel used to compute the Gram matrices.
    Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

The method is implemented by SVD factorization of the weighted 
Gram matrix: 
* D^(1/2) * Phi(X) * Phi(X)' * D^(1/2)
where X is the cenetred matrix and D is a diagonal matrix 
of weights (`weights.w`) of the observations (rows of X).

## References 
Scholkopf, B., Smola, A., MÃ¼ller, K.-R., 1997. Kernel principal 
component analysis, in: Gerstner, W., Germond, A., Hasler, 
M., Nicoud, J.-D. (Eds.), Artificial Neural Networks, 
ICANN 97, Lecture Notes in Computer Science. Springer, Berlin, 
Heidelberg, pp. 583-588. https://doi.org/10.1007/BFb0020217

Scholkopf, B., Smola, A.J., 2002. Learning with kernels: support 
vector machines, regularization, optimization, and beyond, Adaptive 
computation and machine learning. MIT Press, Cambridge, Mass.

Tipping, M.E., 2001. Sparse kernel principal component analysis. Advances 
in neural information processing systems, MIT Press. 
http://papers.nips.cc/paper/1791-sparse-kernel-principal-component-analysis.pdf

## Examples
```julia
using JchemoData, JLD2 
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
pnames(dat)
@head dat.X
X = dat.X[:, 1:4]
n = nro(X)
ntest = 30
s = samprand(n, ntest) 
Xtrain = X[s.train, :]
Xtest = X[s.test, :]

nlv = 3
kern = :krbf ; gamma = 1e-4
mod = kpca(; nlv, kern, 
    gamma) ;
fit!(mod, Xtrain)
pnames(mod.fm)
@head T = mod.fm.T
T' * T
mod.fm.P' * mod.fm.P

@head Ttest = transf(mod, Xtest)

res = summary(mod) ;
pnames(res)
res.explvarx
```
""" 
function kpca(X; kwargs...)
    X = ensure_mat(X)
    Q = eltype(X)
    weights = mweight(ones(Q, nro(X)))
    kpca(X, weights; kwargs...)
end

function kpca(X, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs) 
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
    D = Diagonal(weights.w)
    Kt = K'    
    DKt = D * Kt
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- vtot .+ sum(D * DKt')
    sqrtD = sqrt.(D)
    Kd = sqrtD * Kc * sqrtD
    res = LinearAlgebra.svd(Kd)
    U = res.V[:, 1:nlv]
    eig = res.S
    eig[eig .< 0] .= 0
    sv = sqrt.(eig)
    P = sqrtD * fscale(U, sv[1:nlv])     # In the future: fscale!
    T = Kc * P       # T = Kc * P = D^(-1/2) * U * Delta
    Kpca(X, Kt, T, P, sv, eig, D, DKt, vtot, xscales, 
        weights, kwargs, par)
end

""" 
    transf(object::Kpca, X; nlv = nothing)
Compute PCs (scores T) from a fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which PCs are computed.
* `nlv` : Nb. PCs to compute.
""" 
function transf(object::Kpca, X; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(fscale(X, object.xscales), object.X; object.kwargs...)
    DKt = object.D * K'
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- object.vtot .+ sum(object.D * object.DKt')
    T = Kc * @view(object.P[:, 1:nlv])
    T
end

"""
    summary(object::Kpca)
Summarize the fitted model.
* `object` : The fitted model.
""" 
function Base.summary(object::Kpca)
    nlv = nco(object.T)
    TT = object.D * (object.T).^2
    tt = vec(sum(TT, dims = 1))
    sstot = sum(object.eig)
    pvar = tt / sstot
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(lv = 1:nlv, var = tt, 
        pvar = pvar, cumpvar = cumpvar)
    (explvarx = explvarx,)
end


