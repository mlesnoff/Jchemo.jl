"""
    qda(X, y; kwargs...)
    qda(X, y, weights::Weight; kwargs...)
Quadratic discriminant analysis (QDA, with continuum towards LDA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (the vector must be sorted in the same order as `mlev(y)`).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`) and LDA (`alpha = 1`).

A value `alpha` > 0 shrinks the class-covariances by class 
(Wi) toward a common LDA covariance ("within" W). This corresponds to 
the "first regularization (Eqs.16)" described in Friedman 1989
(where `alpha` is referred to as "lambda").

In these `qda` functions, observation weights (argument `weights`) are used 
to compute covariance matrices Wi and W. Argument `prior` is used to define 
the usual prior class probabilities. 

In the high-level version, the observation weights are automatically 
defined by the given priors (`prior`): the sub-total weights by class are set 
equal to the prior probabilities. For other choices, use the low-level 
version.

## References
Friedman JH. Regularized Discriminant Analysis. Journal 
of the American Statistical Association. 1989; 
84(405):165-175. doi:10.1080/01621459.1989.10478752.

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

mod = model(qda)
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

## With regularization
mod = model(qda; alpha = .5)
#mod = model(qda; alpha = 1) # = LDA
fit!(mod, Xtrain, ytrain)
mod.fm.Wi
res = predict(mod, Xtest) ;
errp(res.pred, ytest)
```
""" 
function qda(X, y; kwargs...)
    par = recovkw(ParQda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    qda(X, y, weights; kwargs...)
end

function qda(X, y, weights::Weight; kwargs...)  
    # Scaling X has no effect
    par = recovkw(ParQda, kwargs).par
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
        if alpha > 0
            @. res.Wi[i] = (1 - alpha) * res.Wi[i] + alpha * res.W
        end
        fm[i] = dmnorm(; mu = ct[i, :], S = res.Wi[i]) 
    end
    Qda(fm, res.Wi, ct, priors, ni, lev, weights, par)
end


    
