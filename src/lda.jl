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
    (in case of vector, it must be sorted in the same order as `mlev(y)`).

In the high-level version of the present `lda` functions, the observation 
weights are automatically defined by the given priors (argument `prior`): 
the sub-total by class of the observation weights are set equal to the prior 
probabilities. The low-level version (argument `weights`) allows to implement 
other choices.

## Examples
```julia
using Jchemo, JchemoData, JLD2
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

mod = model(lda)
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
    par = recovkw(ParLda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    lda(X, y, weights; kwargs...)
end

function lda(X, y, weights::Weight; kwargs...)  
    # Scaling X has no effect
    par = recovkw(ParLda, kwargs).par
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
    @inbounds for i in eachindex(lev)
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(vrow(X, s), mweight(weights.w[s]))
        fm[i] = dmnorm(; mu = ct[i, :], S = res.W) 
    end
    Lda(fm, res.W, ct, priors, ni, lev, weights, par)
end

"""
    predict(object::Union{Lda, Qda}, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Union{Lda, Qda}, X)
    X = ensure_mat(X)
    m = nro(X)
    lev = object.lev
    nlev = length(lev) 
    dens = similar(X, m, nlev)
    @inbounds for i in eachindex(lev)
        dens[:, i] .= vec(predict(object.fm[i], X).pred)
    end
    A = object.priors' .* dens
    v = sum(A, dims = 2)
    posterior = fscale(A', v)'  # Could be replaced by similar as in fscale! 
    z =  mapslices(argmax, posterior; dims = 2)   # if equal, argmax takes the first
    pred = reshape(recod_indbylev(z, lev), m, 1)
    (pred = pred, dens, posterior)
end
    
