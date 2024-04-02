"""
    kdeda(; kwargs...)
    kde(X, y; kwargs...)
Discriminant analysis using non-parametric kernel Gaussian 
    density estimation (KDE-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
Keyword arguments:
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: :unif (uniform; default), 
    :prop (proportional).
* Keyword arguments of function `dmkern` (bandwidth 
    definition) can also be specified here.

The principle is the same as functions `lda` and `qda` 
except that densities are estimated from function `dmkern` 
instead of function `dmnorm`. 

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

prior = :unif
#prior = :prop
mod = kdeda(; prior)
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

mod = kdeda(; prior, a_kde = .5) ;
#mod = kdeda(; prior, h_kde = .1) ;
fit!(mod, Xtrain, ytrain)
mod.fm.fm[1].H
```
""" 
function kdeda(X, y; kwargs...) 
    par = recovkwargs(Par, kwargs)
    X = ensure_mat(X)
    Q = eltype(X)
    lev = mlev(y)
    nlev = length(lev)
    ni = tab(y).vals
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
    for i = 1:nlev
        s = y .== lev[i]
        fm[i] = dmkern(vrow(X, s); h_kde = par.h_kde, a_kde = par.a_kde)
    end
    Kdeda(fm, priors, lev, ni)
end

function predict(object::Kdeda, X)
    X = ensure_mat(X)
    m = nro(X)
    lev = object.lev
    nlev = length(lev) 
    dens = similar(X, m, nlev)
    ni = object.ni
    for i = 1:nlev
        dens[:, i] .= vec(predict(object.fm[i], X).pred)
    end
    A = object.priors' .* dens
    v = sum(A, dims = 2)
    posterior = fscale(A', v)'    # Could be replaced by similar as in fscale! 
    z =  mapslices(argmax, posterior; dims = 2)    # if equal, argmax takes the first
    pred = reshape(replacebylev2(z, object.lev), m, 1)
    (pred = pred, dens, posterior)
end
    
