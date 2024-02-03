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
    `:prop` (proportional).

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

res = predict(mod, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
confusion(res.pred, ytest).cnt
```
""" 
function lda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    lda(X, y, weights; kwargs...)
end

function lda(X, y, weights::Weight; kwargs...)  
    par = recovkwargs(Par, kwargs)
    @assert in([:unif; :prop])(par.prior) "Wrong value for argument 'prior'."
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
    if isequal(par.prior, :unif)
        wprior = ones(Q, nlev) / nlev
    elseif isequal(par.prior, :prop)
        wprior = convert.(Q, mweight(ni).w)
    end
    fm = list(nlev)
    ct = similar(X, nlev, p)
    @inbounds for i = 1:nlev
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(X[s, :], mweight(weights.w[s]))
        fm[i] = dmnorm(; mu = ct[i, :], S = res.W) 
    end
    Lda(fm, res.W, ct, wprior, res.theta.w, ni, lev, weights)
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
        dens[:, i] .= vec(predict(object.fm[i], 
            X).pred)
    end
    A = object.wprior' .* dens
    v = sum(A, dims = 2)
    posterior = fscale(A', v)'  # Could be replaced by similar as in fscale! 
    z =  mapslices(argmax, posterior; dims = 2)   # if equal, argmax takes the first
    pred = reshape(replacebylev2(z, lev), m, 1)
    (pred = pred, dens, posterior)
end
    
