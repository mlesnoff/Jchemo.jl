"""
    kpca(X, weights = ones(nro(X)); nlv, 
        kern = :krbf, scal::Bool = false, kwargs...)
Kernel PCA  (Scholkopf et al. 1997, Scholkopf & Smola 2002, Tipping 2001).

* `X` : X-data.
* `weights` : vector (n,).
* `nlv` : Nb. principal components (PCs), or collection of nb. PCs, to consider. 
* `kern` : Type of kernel used to compute the Gram matrices.
    Possible values are :krbf of :kpol (see respective 
    functions `krbf` and `kpol`).
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.
* `kwargs` : Named arguments to pass in the kernel function. 
    See `?krbf`, `?kpol`.

The method is implemented by SVD factorization of the weighted Gram matrix 
D^(1/2) * Phi(`X`) * Phi(`X`)' * D^(1/2), where D is a diagonal matrix of weights for 
the observations (rows of X).

The kernel Gram matrices are internally centered. 

## References 
Scholkopf, B., Smola, A., MÃ¼ller, K.-R., 1997. Kernel principal component analysis, 
in: Gerstner, W., Germond, A., Hasler, M., Nicoud, J.-D. (Eds.), Artificial Neural Networks, 
ICANN 97, Lecture Notes in Computer Science. Springer, Berlin, Heidelberg, 
pp. 583-588. https://doi.org/10.1007/BFb0020217

Scholkopf, B., Smola, A.J., 2002. Learning with kernels: support vector machines, regularization, 
optimization, and beyond, Adaptive computation and machine learning. MIT Press, Cambridge, Mass.

Tipping, M.E., 2001. Sparse kernel principal component analysis. Advances in neural information 
processing systems, MIT Press. http://papers.nips.cc/paper/1791-sparse-kernel-principal-component-analysis.pdf

## Examples
```julia
using JchemoData, JLD2, CairoMakie, StatsBase
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
pnames(dat)
summ(dat.X)

X = dat.X[:, 1:4]
n = nro(X)

ntrain = 120
s = sample(1:n, ntrain; replace = false) 
Xtrain = X[s, :]
Xtest = rmrow(X, s)

nlv = 3 ; gamma = 1e-4
fm = kpca(Xtrain; nlv = nlv, gamma = gamma) ;
pnames(fm)
fm.T
fm.T' * fm.T
fm.P' * fm.P

Jchemo.transform(fm, Xtest)

res = Base.summary(fm) ;
pnames(res)
res.explvarx
```
""" 
function kpca(X, weights = ones(nro(X)); nlv, 
        kern = :krbf, scal::Bool = false, kwargs...)
    X = ensure_mat(X)
    n, p = size(X)
    nlv = min(nlv, n)
    weights = mweight(weights) 
    xscales = ones(eltype(X), p)
    if par.scal 
        xscales .= colstd(X, weights)
        X = scale(X, xscales)
    end
    fkern = eval(Meta.parse(String(par.kern)))  
    K = fkern(X, X; kwargs...)     # In the future: fkern!(K, X, X; kwargs...)
    D = Diagonal(weights)
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
    P = sqrtD * scale(U, sv[1:nlv])     # In the future: scale!
    T = Kc * P       # T = Kc * P = D^(-1/2) * U * Delta
    Kpca(X, Kt, T, P, sv, eig, D, DKt, vtot, xscales, weights, kern, kwargs)
end

""" 
    transform(object::Kpca, X; nlv = nothing)
Compute PCs (scores T) from a fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which PCs are computed.
* `nlv` : Nb. PCs to consider.
""" 
function transform(object::Kpca, X; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(scale(X, object.xscales), object.X; par = object.par)
    DKt = object.D * K'
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- object.vtot .+ sum(object.D * object.DKt')
    T = Kc * @view(object.P[:, 1:nlv])
    T
end

"""
    summary(object::Kpca, X)
Summarize the maximal (i.e. with maximal nb. PCs) fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
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


