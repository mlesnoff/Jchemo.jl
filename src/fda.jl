"""
    fda(X, y; nlv, lb = 0, scal::Bool = false)
    fda!(X::Matrix, y; nlv, lb = 0, scal::Bool = false)
Factorial discriminant analysis (FDA).
* `X` : X-data (n, p).
* `y` : y-data (n) (class membership).
* `nlv` : Nb. discriminant components.
* `lb` : Ridge regularization parameter "lambda".
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

FDA by eigen factorization of Inverse(W) * B, where W is the 'Within'-covariance 
matrix (pooled over the classes), and B the 'Between'-covariance matrix.

The function maximizes the compromise p'Bp / p'Wp, i.e. max p'Bp with 
constraint p'Wp = 1. Vectors p (columns of P) are the linear discrimant 
coefficients often referred to as "LD".

A ridge regularization can be used:
* If `lb` > 0, W is replaced by W + `lb` * I, 
    where I is the Idendity matrix.

## Examples
```julia
using JchemoData, JLD2, StatsBase, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
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
    ellipse = true, title = "FDA")
scatter!(ax, ct[:, 1], ct[:, 2];  
    markersize = 10, color = :red)
hlines!(ax, 0; color = :grey)
vlines!(ax, 0; color = :grey)
f

# Projection of Xtest to the score space
Jchemo.transform(fm, Xtest)

# X-loadings matrix
# Columns of P = coefficients of the linear discriminant function
# = "LD" of function lda of R package MASS
fm.P
fm.P' * fm.P    # not orthogonal

fm.eig
fm.sstot
# Explained variance by PCA of the class centers 
# in transformed scale
Base.summary(fm)
```
""" 
function fda(X, y; nlv, lb = 0, scal::Bool = false)
    fda!(copy(ensure_mat(X)), y; nlv = nlv, lb = lb, 
        scal = scal)
end

function fda!(X::Matrix, y; nlv, lb = 0, scal::Bool = false)
    n, p = size(X)
    xmeans = colmean(X)
    xscales = ones(eltype(X), p)
    if par.scal 
        xscales .= colstd(X)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end    
    res = matW(X, y)
    lev = res.lev
    nlev = length(lev)
    ni = res.ni
    res.W .*= n / (n - nlev)    # unbiased estimate
    if lb > 0
        res.W .= res.W .+ lb .* I(p)    # @. does not work with I
    end
    zres = matB(X, y)
    Winv = LinearAlgebra.inv!(cholesky(Hermitian(res.W))) 
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
    summary(object::Fda, X)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Fda)
    nlv = nco(object.T)
    eig = object.eig[1:nlv]
    pvar =  eig ./ sum(object.eig)
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(lv = 1:nlv, var = eig, pvar = pvar, 
        cumpvar = cumpvar)
    (explvarx = explvarx,)    
end

