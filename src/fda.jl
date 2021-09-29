"""
    fda(X, y; nlv, pseudo = false)
Factorial discriminant analysis (FDA).
* `X` : X-data.
* `y` : Univariate class membership.
* `nlv` : Nb. discriminant components.
* `pseudo` : If true, a MP pseudo-inverse is used (instead
    of a usual inverse) for inverting W.

Eigen factorization of Inverse(W)*B. 

The functions maximize the compromise p'Bp / p'Wp, i.e. max p'Bp with 
constraint p'Wp = 1. Vectors p (columns of P) are the linear discrimant 
coefficients "LD".

`X` is internally centered.  
""" 
function fda(X, y; nlv, pseudo = false)
    fda!(copy(X), y; nlv = nlv, pseudo = pseudo)
end

function fda!(X, y; nlv, pseudo = false)
    X = ensure_mat(X)
    n, p = size(X)
    nlv = min(nlv, n, p)
    xmeans = colmeans(X) 
    center!(X, xmeans)
    z = matW(X, y)
    lev = z.lev
    nlev = length(lev)
    W = z.W * n / (n - nlev)
    ni = z.ni
    z = matB(X, y)
    B = z.B
    ct = z.ct
    nlv = min(nlv, p, nlev - 1)
    pseudo ? Winv = pinv(W) : Winv = inv(W)
    # Winv * B is not symmetric
    res = eigen!(Winv * B; sortby = x -> -abs(x))
    P = res.vectors[:, 1:nlv]
    eig = res.values
    P = real.(P)
    eig = real.(eig)
    sstot = sum(eig)
    norm_P = sqrt.(diag(P' * W * P))
    scale!(P, norm_P)
    T = X * P
    Tcenters = ct * P
    (T = T, P = P, Tcenters = Tcenters, eig = eig, sstot = sstot,
        W = W, xmeans = xmeans, lev = lev, ni = ni)
end

function fdasvd(X, y; nlv, pseudo = false)
    fdasvd!(copy(X), y; nlv = nlv, pseudo = pseudo)
end

"""
    fda(X, y; nlv, pseudo = false)
Factorial discriminant analysis (FDA).
* `X` : X-data.
* `y` : Univariate class membership.
* `nlv` : Nb. discriminant components.
* `pseudo` : If true, a MP pseudo-inverse is used (instead
    of a usual inverse) for inverting W.

Weighted SVD factorization of the matrix of the class centers.

`X` is internally centered.  
""" 
function fdasvd!(X, y; nlv, pseudo = false)
    X = ensure_mat(X)
    n, p = size(X)
    nlv = min(nlv, n, p)
    xmeans = colmeans(X) 
    center!(X, xmeans)
    z = matW(X, y)
    lev = z.lev
    nlev = length(lev)
    W = z.W * n / (n - nlev)
    ni = z.ni
    z = matB(X, y)
    B = z.B
    ct = z.ct
    nlv = min(nlv, p, nlev - 1)
    pseudo ? Winv = pinv(W) : Winv = inv(W)
    U = cholesky!(Hermitian(Winv))
    Ut = U'
    Zct = ct * Ut
    zfm = pcasvd(Zct, ni; nlv = nlev - 1)
    Pz = zfm.P
    Tcenters = Zct * Pz        
    eig = sqrt.(zfm.sv)
    sstot = sum(eig)
    P = Ut * Pz[:, 1:nlv]
    T = X * P
    Tcenters = ct * P
    (T = T, P = P, Tcenters = Tcenters, eig = eig, sstot = sstot,
        W = W, xmeans = xmeans, lev = lev, ni = ni)
end









