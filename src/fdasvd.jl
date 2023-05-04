"""
    fdasvd(X, y; nlv, pseudo = false, scal = false)
    fdasvd!(X, y; nlv, pseudo = false, scal = false)
Factorial discriminant analysis (FDA).
* `X` : X-data.
* `y` : Univariate class membership.
* `nlv` : Nb. discriminant components.
* `pseudo` : If true, a MP pseudo-inverse is used (instead
    of a usual inverse) for inverting W.
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

Weighted SVD factorization of the matrix of the class centers.

See `?fda` for examples.

""" 
function fdasvd(X, y; nlv, pseudo = false, scal = false)
    fdasvd!(copy(ensure_mat(X)), y; nlv = nlv, pseudo = pseudo, 
        scal = scal)
end

function fdasvd!(X::Matrix, y; nlv, pseudo = false, scal = false)
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
    res.W .= res.W * n / (n - nlev)
    !pseudo ? Winv = inv(res.W) : Winv = pinv(res.W)
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
