"""
    fda(; kwargs...)
    fda(X, y; kwargs...)
    fda!(X::Matrix, y; kwargs...)
Factorial discriminant analysis (FDA).
* `X` : X-data (n, p).
* `y` : y-data (n) (class membership).
Keyword arguments:
* `nlv` : Nb. of discriminant components.
* `lb` : Ridge regularization parameter "lambda".
    Can be used when `X` has collinearities. 
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

FDA by eigen factorization of Inverse(W) * B, where W is 
the 'Within'-covariance matrix (pooled over the classes), 
and B the 'Between'-covariance matrix.

The function maximizes the compromise:0
* p'Bp / p'Wp 
i.e. max p'Bp with constraint p'Wp = 1. Vectors p 
(columns of `P`) are the linear discrimant coefficients 
often referred to as "LD".

If `X` is ill-conditionned, a ridge regularization can 
be used:
* If `lb` > 0, W is replaced by W + `lb` * I, 
    where I is the Idendity matrix.

## Examples
```julia
using JchemoData, JLD2, CairoMakie 
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
pnames(dat)
@head dat.X
X = dat.X[:, 1:4]
y = dat.X[:, 5]
n = nro(X)
ntest = 30
s = samprand(n, ntest) 
Xtrain = X[s.train, :]
ytrain = y[s.train]
Xtest = X[s.test, :]
ytest = y[s.test]
tab(ytrain)
tab(ytest)

nlv = 2 ; scal = false
mod = fda(; nlv, scal)
#mod = fdasvd(; nlv, scal)
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
lev = fm.lev
nlev = length(lev)
@head fm.T 
@head transf(mod, Xtrain)
@head transf(mod, Xtest)

## X-loadings matrix
## = coefficients of the linear discriminant function
## = "LD" of function lda of the R package MASS
fm.P
fm.P' * fm.P

## Explained variance computed by weighted PCA 
## of the class centers in transformed scale
summary(mod).explvarx

## Projections of the class centers 
## to the score space
ct = fm.Tcenters 
f, ax = plotxy(fm.T[:, 1], fm.T[:, 2], ytrain;
    xlabel = "Score-1", ylabel = "Score-2",
    title = "FDA")
scatter!(ax, ct[:, 1], ct[:, 2], 
    markersize = 15, color = :red)
f
```
""" 
fda(X, y; kwargs...) = fda!(copy(ensure_mat(X)), y; 
    kwargs...)

function fda!(X::Matrix, y; kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert par.lb >= 0 "Argument 'lb' must ∈ [0, Inf[."
    Q = eltype(X)
    n, p = size(X)
    lb = convert(Q, par.lb)
    xmeans = colmean(X)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    w = mweight(ones(Q, n))    
    res = matW(X, y, w)
    lev = res.lev
    nlev = length(lev)
    ni = res.ni
    res.W .*= n / (n - nlev)    # unbiased estimate
    if lb > 0
        res.W .= res.W .+ lb .* I(p)    # @. does not work with I
    end
    zres = matB(X, y, w)
    Winv = LinearAlgebra.inv!(cholesky(Hermitian(res.W))) 
    ## Winv * B is not symmetric
    fm = eigen!(Winv * zres.B; sortby = x -> -abs(x))
    nlv = min(par.nlv, n, p, nlev - 1)
    P = fm.vectors[:, 1:nlv]
    eig = fm.values
    P = real.(P)
    eig = real.(eig)
    sstot = sum(eig)
    norm_P = sqrt.(diag(P' * res.W * P))
    fscale!(P, norm_P)
    T = X * P
    Tcenters = zres.ct * P
    Fda(T, P, Tcenters, eig, sstot, res.W, xmeans, 
        xscales, lev, ni, kwargs, par)
end

"""
    summary(object::Fda)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Fda)
    nlv = nco(object.T)
    eig = object.eig[1:nlv]
    pvar =  eig ./ sum(object.eig)
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(lv = 1:nlv, var = eig, 
        pvar = pvar, cumpvar = cumpvar)
    (explvarx = explvarx,)    
end

