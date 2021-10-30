struct Fda
    T::Array{Float64}
    P::Array{Float64}
    Tcenters::Array{Float64}
    eig::Vector{Float64}
    sstot::Number
    W::Matrix{Float64}
    xmeans::Vector{Float64}
    lev::AbstractVector
    ni::AbstractVector
end

"""
    fda(X, y; nlv, pseudo = false)
Factorial discriminant analysis (FDA).
* `X` : X-data.
* `y` : Univariate class membership.
* `nlv` : Nb. discriminant components.
* `pseudo` : If true, a MP pseudo-inverse is used (instead
    of a usual inverse) for inverting W.

Eigen factorization of Inverse(W) * B. 

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
    xmeans = colmeans(X) 
    center!(X, xmeans)
    res = matW(X, y)
    lev = res.lev
    nlev = length(lev)
    ni = res.ni
    res.W .= res.W * n / (n - nlev)
    zres = matB(X, y)
    !pseudo ? Winv = LinearAlgebra.inv!(cholesky!(Hermitian(res.W))) : Winv = pinv(res.W)
    # Winv * B is not symmetric
    fm = eigen!(Winv * zres.B; sortby = x -> -abs(x))
    nlv = min(nlv, n, p, nlev - 1)
    P = fm.vectors[:, 1:nlv]
    eig = fm.values
    P = real.(P)
    eig = real.(eig)
    sstot = sum(eig)
    norm_P = sqrt.(diag(P' * res.W * P))
    scale!(P, norm_P)
    T = X * P
    Tcenters = zres.ct * P
    Fda(T, P, Tcenters, eig, sstot, res.W, xmeans, lev, ni)
end

function fdasvd(X, y; nlv, pseudo = false)
    fdasvd!(copy(X), y; nlv = nlv, pseudo = pseudo)
end

"""
    fdasvd(X, y; nlv, pseudo = false)
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
    xmeans = colmeans(X) 
    center!(X, xmeans)
    res = matW(X, y)
    lev = res.lev
    nlev = length(lev)
    ni = res.ni
    res.W .= res.W * n / (n - nlev)
    !pseudo ? Winv = inv(res.W) : Winv = pinv(res.W)
    ct = aggstat(X, y; fun = mean).res
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
    Fda(T, P, Tcenters, eig, sstot, res.W, xmeans, lev, ni)
end

"""
    summary(object::Fda, X)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Fda)
    nlv = size(object.T, 2)
    eig = object.eig[1:nlv]
    pvar =  eig ./ sum(object.eig)
    cumpvar = cumsum(pvar)
    explvar = DataFrame(lv = 1:nlv, var = eig, pvar = pvar, 
        cumpvar = cumpvar)
    (explvar = explvar,)    
end




