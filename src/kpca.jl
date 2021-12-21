struct Kpca
    X::Array{Float64}
    Kt::Array{Float64}
    T::Array{Float64}
    P::Array{Float64}
    sv::Vector{Float64}  
    eig::Vector{Float64}    
    D::Array{Float64} 
    DKt::Array{Float64}
    vtot::Array{Float64} 
    weights::Vector{Float64}
    kern
    dots
end

"""
    kpca(X, Y, weights = ones(size(X, 1)); nlv , kern = "krbf", kwargs...)
Kernel PCA  (Scholkopf et al. 1997, Scholkopf & Smola 2002, Tipping 2001).

* `X` : X-data.
* `weights` : vector (n,).
* `nlv` : Nb. principal components (PCs), or collection of nb. PCs, to consider. 
* `kern` : Type of kernel used to compute the Gram matrices.
    Possible values are "krbf" of "kpol" (see respective functions `krbf` and `kpol`).
* `kwargs` : Named arguments to pass in the kernel function.

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

""" 
function kpca(X, weights = ones(size(X, 1)); nlv, kern = "krbf", kwargs...)
    X = ensure_mat(X)
    n = size(X, 1)
    nlv = min(nlv, n)
    weights = mweights(weights) 
    fkern = eval(Meta.parse(kern))  
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
    Kpca(X, Kt, T, P, sv, eig, D, DKt, vtot, weights, kern, kwargs)
end

"""
    summary(object::Kpca, X)
Summarize the maximal (i.e. with maximal nb. PCs) fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Kpca)
    nlv = size(object.T, 2)
    TT = object.D * (object.T).^2
    tt = vec(sum(TT, dims = 1))
    sstot = sum(object.eig)
    pvar = tt / sstot
    cumpvar = cumsum(pvar)
    explvar = DataFrame(pc = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    (explvar = explvar,)
end

""" 
    transform(object::Kpca, X; nlv = nothing)
Compute PCs (scores matrix "T") from a fitted model and X-data.
* `object` : The maximal fitted model.
* `X` : X-data for which PCs are computed.
* `nlv` : Nb. PCs to consider. If nothing, it is the maximum nb. PCs.
""" 
function transform(object::Kpca, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    fkern = eval(Meta.parse(object.kern))
    K = fkern(X, object.X; object.dots...)
    DKt = object.D * K'
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- object.vtot .+ sum(object.D * object.DKt')
    T = Kc * @view(object.P[:, 1:nlv])
    T
end






