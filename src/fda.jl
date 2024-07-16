"""
    fda(X, y; kwargs...)
    fda(X, y, weights; kwargs...)
    fda!(X::Matrix, y, weights; kwargs...)
Factorial discriminant analysis (FDA).
* `X` : X-data (n, p).
* `y` : y-data (n) (class membership).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. of discriminant components.
* `lb` : Ridge regularization parameter "lambda".
    Can be used when `X` has collinearities. 
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (the vector must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

FDA by eigen factorization of Inverse(W) * B, where W is 
the "Within"-covariance matrix (pooled over the classes), 
and B the "Between"-covariance matrix.

The function maximizes the compromise:
* p'Bp / p'Wp 
i.e. max p'Bp with constraint p'Wp = 1. Vectors p (columns of `P`) 
are the linear discrimant coefficients often referred to as "LD".

If `X` is ill-conditionned, a ridge regularization can be used:
* If `lb` > 0, W is replaced by W + `lb` * I, 
    where I is the Idendity matrix.

In these `fda` functions, observation weights (argument `weights`) are used 
to compute matrices W and B. 

In the high-level version, the observation weights are automatically 
defined by the given priors (argument `prior`): the sub-total weights by 
class are set equal to the prior probabilities. For other choices, 
use the low-level versions.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie 
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

nlv = 2
mod = model(fda; nlv)
#mod = model(fdasvd; nlv)
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
lev = fm.lev
nlev = length(lev)
aggsum(fm.weights.w, ytrain)

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
f, ax = plotxy(fm.T[:, 1], fm.T[:, 2], ytrain; ellipse = true, title = "FDA",
    xlabel = "Score-1", ylabel = "Score-2")
scatter!(ax, ct[:, 1], ct[:, 2], marker = :star5, markersize = 15, color = :red)  # see available_marker_symbols()
f
```
""" 
function fda(X, y; kwargs...)
    par = recovkw(Par, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    fda(X, y, weights; kwargs...)
end

fda(X, y, weights; kwargs...) = fda!(copy(ensure_mat(X)), y, weights; kwargs...)

function fda!(X::Matrix, y, weights; kwargs...)
    par = recovkw(Par, kwargs).par
    @assert par.lb >= 0 "Argument 'lb' must ∈ [0, Inf[."
    Q = eltype(X)
    n, p = size(X)
    lb = convert(Q, par.lb)
    xmeans = colmean(X, weights)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    res = matW(X, y, weights)
    lev = res.lev
    nlev = length(lev)
    ni = res.ni
    res.W .*= n / (n - nlev)    # unbiased estimate
    if lb > 0
        res.W .+= lb .* I(p)    # @. does not work with I
    end
    zres = matB(X, y, weights)
    Winv = LinearAlgebra.inv!(cholesky(Hermitian(res.W)))
    ## Winv * B is not symmetric
    fm = eigen!(Winv * zres.B; sortby = x -> -abs(x))
    nlv = min(par.nlv, n, p, nlev - 1)
    P = real.(fm.vectors[:, 1:nlv])
    eig = real.(fm.values)
    sstot = sum(eig)
    norm_P = sqrt.(diag(P' * res.W * P))
    fscale!(P, norm_P)
    T = X * P
    Tcenters = zres.ct * P
    Fda(T, P, Tcenters, eig, sstot, res.W, xmeans, xscales, weights, 
        lev, ni, par)
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
    explvarx = DataFrame(lv = 1:nlv, var = eig, pvar = pvar, cumpvar = cumpvar)
    (explvarx = explvarx,)    
end

