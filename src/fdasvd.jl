"""
    fdasvd(X, y; nlv, lb = 0, scal = false)
    fdasvd!(X, y; nlv, lb = 0, scal = false)
Factorial discriminant analysis (FDA).
* `X` : X-data.
* `y` : Univariate class membership.
* `nlv` : Nb. discriminant components.
* `lb` : Ridge regularization parameter "lambda".
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

FDA by weighted SVD factorization of the matrix of the class centers.

A ridge regularization can be used:
* If `lb` > 0, the within-class (pooled) covariance matrix W 
    is replaced by W + `lb` * I, where I is the Idendity matrix.

See `?fda` for examples.

""" 
function fdasvd(X, y; nlv, lb = 0, scal = false)
    fdasvd!(copy(ensure_mat(X)), y; nlv = nlv, lb = lb, 
        scal = scal)
end

function fdasvd!(X::Matrix, y; nlv, lb = 0, scal = false)
    n, p = size(X)
    xmeans = colmean(X) 
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end
    res = matW(X, y)
    lev = res.lev
    nlev = length(lev)
    ni = res.ni
    res.W .*= n / (n - nlev)
    if lb > 0
        res.W .= res.W .+ (lb * I(p)) # @. does not work with I
    end
    #Winv = inv(W)
    Winv = LinearAlgebra.inv!(cholesky(Hermitian(res.W))) 
    ct = aggstat(X, y; fun = mean).X
    Ut = cholesky!(Hermitian(Winv)).U'
    Zct = ct * Ut
    nlv = min(nlv, n, p, nlev - 1)
    fm = pcasvd(Zct, ni; nlv = nlv)
    Pz = fm.P
    Tcenters = Zct * Pz        
    eig = (fm.sv).^2 
    sstot = sum(eig)
    P = Ut * Pz[:, 1:nlv]
    T = X * P
    Tcenters = ct * P
    Fda(T, P, Tcenters, eig, sstot, res.W, xmeans, xscales, lev, ni)
end
