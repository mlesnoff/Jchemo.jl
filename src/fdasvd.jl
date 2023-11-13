"""
    fdasvd(X, y; nlv, lb = 0, scal::Bool = false)
    fdasvd!(X, y; nlv, lb = 0, scal::Bool = false)
Factorial discriminant analysis (FDA).
* `X` : X-data.
* `y` : Univariate class membership.
* `nlv` : Nb. discriminant components.
* `lb` : Ridge regularization parameter "lambda".
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

FDA by a weighted SVD factorization of the matrix of the class 
centers (after spherical transformaton). 
The function gives the same results as function `fda`.

A ridge regularization can be used:
* If `lb` > 0, the within-class (pooled) covariance matrix W 
    is replaced by W + `lb` * I, where I is the Idendity matrix.

See `?fda` for examples.

""" 
fdasvd(X, y; par = Par()) = fdasvd!(copy(ensure_mat(X)), y; par)

function fdasvd!(X::Matrix, y; par = Par())
    @assert par.lb >= 0 "Argument 'lb' must ∈ [0, Inf[."
    n, p = size(X)
    lb = convert(eltype(X), par.lb)
    xmeans = colmean(X) 
    xscales = ones(eltype(X), p)
    if par.scal 
        xscales .= colstd(X)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end
    w = mweight(ones(eltype(X), n))
    res = matW(X, y, w)
    lev = res.lev
    nlev = length(lev)
    ni = res.ni
    res.W .*= n / (n - nlev)
    if lb > 0
        res.W .= res.W .+ lb .* I(p) # @. does not work with I
    end
    #Winv = inv(res.W)
    Winv = LinearAlgebra.inv!(cholesky(Hermitian(res.W))) 
    ct = aggstat(X, y; fun = mean).X
    Ut = cholesky!(Hermitian(Winv)).U'
    Zct = ct * Ut
    nlv = min(par.nlv, n, p, nlev - 1)
    par = Par(nlv = nlv, scal = false)
    fm = pcasvd(Zct, mweight(ni); par)
    Pz = fm.P
    Tcenters = Zct * Pz        
    eig = (fm.sv).^2 
    sstot = sum(eig)
    P = Ut * Pz[:, 1:nlv]
    T = X * P
    Tcenters = ct * P
    Fda(T, P, Tcenters, eig, sstot, res.W, xmeans, xscales, lev, ni)
end
