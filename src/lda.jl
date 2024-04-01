"""
    lda(; kwargs...)
    lda(X, y; kwargs...)
    lda(X, y, weights::Weight; kwargs...)
Linear discriminant analysis (LDA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (the vector must be sorted in the same order as `mlev(x)`).

In these functions, observation weights (argument `weights`) are used 
to compute the intra-class ("within") covariance matrix. Argument `prior` 
is used to define the prior class probabilities. 

In the high-level version of the functions, the observation 
weights are automatically defined by the given priors: the sub-total 
weights by class are set equal to the prior probabilities. 
For more generality, use the low-level version.

## Examples
```julia
using JchemoData, JLD2
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
ntrain = n - ntest
(ntot = n, ntrain, ntest)
tab(ytrain)
tab(ytest)

mod = lda()
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
fm.lev
fm.ni
aggsum(fm.weights.w, ytrain)

res = predict(mod, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt
```
""" 
function lda(X, y; kwargs...)
    par = recovkwargs(Par, kwargs)
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    lda(X, y, weights; kwargs...)
end

function lda(X, y, weights::Weight; kwargs...)  
    par = recovkwargs(Par, kwargs)
    # Scaling X has no effect
    X = ensure_mat(X)
    y = vec(y)    # for findall
    Q = eltype(X)
    n, p = size(X)
    res = matW(X, y, weights)
    ni = res.ni
    lev = res.lev
    nlev = length(lev)
    res.W .*= n / (n - nlev)    # unbiased estimate
    ## Priors
    if isequal(par.prior, :unif)
        priors = ones(Q, nlev) / nlev
    elseif isequal(par.prior, :prop)
        priors = convert.(Q, mweight(ni).w)
    else
        priors = mweight(par.prior).w
    end
    ## End
    fm = list(nlev)
    ct = similar(X, nlev, p)
    @inbounds for i = 1:nlev
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(vrow(X, s), mweight(weights.w[s]))
        fm[i] = dmnorm(; mu = ct[i, :], S = res.W) 
    end
    Lda(fm, res.W, ct, priors, ni, lev, weights)
end

"""
    predict(object::Lda, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lda, X)
    X = ensure_mat(X)
    m = nro(X)
    lev = object.lev
    nlev = length(lev) 
    dens = similar(X, m, nlev)
    for i = 1:nlev
        dens[:, i] .= vec(predict(object.fm[i], X).pred)
    end
    A = object.priors' .* dens
    v = sum(A, dims = 2)
    posterior = fscale(A', v)'  # Could be replaced by similar as in fscale! 
    z =  mapslices(argmax, posterior; dims = 2)   # if equal, argmax takes the first
    pred = reshape(replacebylev2(z, lev), m, 1)
    (pred = pred, dens, posterior)
end
    
