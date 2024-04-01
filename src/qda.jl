"""
    qda(; kwargs...)
    qda(X, y; kwargs...)
    qda(X, y, weights::Weight; kwargs...)
Quadratic discriminant analysis (QDA, with continuum 
    towards LDA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`) and LDA (`alpha = 1`).

A value `alpha` > 0 shrinks the class-covariances by class 
(Wi) toward a common LDA covariance (W). This corresponds to 
the "first regularization (Eqs.16)" described in Friedman 1989
(where `alpha` is referred to as "lambda" in Friedman 1989).

## References
Friedman JH. Regularized Discriminant Analysis. Journal 
of the American Statistical Association. 1989; 
84(405):165-175. doi:10.1080/01621459.1989.10478752.

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

mod = qda()
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
conf(res.pred, ytest).cnt

## With regularization
mod = qda(; alpha = .5)
#mod = qda(; alpha = 1) # = LDA
fit!(mod, Xtrain, ytrain)
mod.fm.Wi
res = predict(mod, Xtest) ;
errp(res.pred, ytest)
```
""" 
function qda(X, y; kwargs...)
    par = recovkwargs(Par, kwargs)
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    qda(X, y, weights; kwargs...)
end

function qda(X, y, weights::Weight; kwargs...)  
    # Scaling X has no effect
    par = recovkwargs(Par, kwargs)
    @assert 0 <= par.alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."
    X = ensure_mat(X)
    y = vec(y)    # for findall
    Q = eltype(X)
    n, p = size(X)
    alpha = convert(Q, par.alpha)
    res = matW(X, y, weights)
    ni = res.ni
    lev = res.lev
    nlev = length(lev)
    res.W .*= n / (n - nlev)    # unbiased estimate
    if isequal(par.prior, :unif)
        priors = ones(Q, nlev) / nlev
    elseif isequal(par.prior, :prop)
        priors = convert.(Q, mweight(ni).w)
    end
    fm = list(nlev)
    ct = similar(X, nlev, p)
    @inbounds for i = 1:nlev
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(X[s, :], mweight(weights.w[s]))
        ni[i] == 1 ? zn = n : zn = ni[i]
        res.Wi[i] .*= zn / (zn - 1)
        if alpha > 0
            @. res.Wi[i] = (1 - alpha) * res.Wi[i] + alpha * res.W
        end
        fm[i] = dmnorm(; mu = ct[i, :], S = res.Wi[i]) 
    end
    Qda(fm, res.Wi, ct, priors, ni, lev, weights)
end

"""
    predict(object::Qda, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Qda, X)
    X = ensure_mat(X)
    m = nro(X)
    lev = object.lev
    nlev = length(lev) 
    dens = similar(X, m, nlev)
    for i = 1:nlev
        dens[:, i] .= vec(Jchemo.predict(object.fm[i], X).pred)
    end
    A = object.priors' .* dens
    v = sum(A, dims = 2)
    posterior = fscale(A', v)'  # Could be replaced by similar as in fscale! 
    z =  mapslices(argmax, posterior; dims = 2)  # if equal, argmax takes the first
    pred = reshape(replacebylev2(z, lev), m, 1)
    (pred = pred, dens, posterior)
end
    
