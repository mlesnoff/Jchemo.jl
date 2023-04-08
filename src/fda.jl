struct Fda
    T::Array{Float64}
    P::Array{Float64}
    Tcenters::Array{Float64}
    eig::Vector{Float64}
    sstot::Number
    W::Matrix{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    lev::AbstractVector
    ni::AbstractVector
end

"""
    fda(X, y; nlv, pseudo = false, scal = false)
    fda!(X::Matrix, y; nlv, pseudo = false, scal = false)
Factorial discriminant analysis (FDA).
* `X` : X-data (n, p).
* `y` : y-data (n) (class membership).
* `nlv` : Nb. discriminant components.
* `pseudo` : If true, a MP pseudo-inverse is used (instead
    of a usual inverse) for inverting W.
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

Eigen factorization of Inverse(W) * B. 

The functions maximize the compromise p'Bp / p'Wp, i.e. max p'Bp with 
constraint p'Wp = 1. Vectors p (columns of P) are the linear discrimant 
coefficients "LD".

## Examples
```julia
using Jchemo, JLD2, StatsBase, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data", "iris.jld2") 
@load db dat
pnames(dat)
summ(dat.X)

X = dat.X[:, 1:4] 
y = dat.X[:, 5]
n = nro(X)

ntrain = 120
s = sample(1:n, ntrain; replace = false) 
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

tab(ytrain)
tab(ytest)

fm = fda(Xtrain, ytrain; nlv = 2) ;
#fm = fdasvd(Xtrain, ytrain; nlv = 2) ;
pnames(fm)
lev = fm.lev
nlev = length(lev)

fm.T
# Projections of the class centers to the score space
ct = fm.Tcenters 

group = copy(ytrain)
f, ax = plotxy(fm.T[:, 1], fm.T[:, 2], group;
    title = "FDA")
scatter!(ax, ct[:, 1], ct[:, 2], 
    markersize = 10, color = :red)
f

# Projection of Xtest to the score space
Jchemo.transform(fm, Xtest)

# X-loadings matrix
# = coefficients of the linear discriminant function
# = "LD" of function lda of package MASS
fm.P

fm.eig
fm.sstot
# Explained variance by PCA of the class centers 
# in transformed scale
Base.summary(fm)
```
""" 
function fda(X, y; nlv, pseudo = false, scal = false)
    fda!(copy(ensure_mat(X)), y; nlv = nlv, pseudo = pseudo, 
        scal = scal)
end

function fda!(X::Matrix, y; nlv, pseudo = false, scal = false)
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
    zres = matB(X, y)
    if !pseudo 
        Winv = LinearAlgebra.inv!(cholesky!(Hermitian(res.W)))
    else 
        Winv = pinv(res.W)
    end
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
    Fda(T, P, Tcenters, eig, sstot, res.W, xmeans, xscales, lev, ni)
end

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
    ni = res.ni
    nlev = length(lev)
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
    explvarx = DataFrame(lv = 1:nlv, var = eig, pvar = pvar, 
        cumpvar = cumpvar)
    (explvarx = explvarx,)    
end

