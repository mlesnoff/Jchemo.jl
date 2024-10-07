"""
    kdeda(X, y; kwargs...)
Discriminant analysis using non-parametric kernel Gaussian 
    density estimation (KDE-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
Keyword arguments:
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* Keyword arguments of function `dmkern` (bandwidth 
    definition) can also be specified here.

The principle is the same as functions `qda` except that densities by class 
are estimated from function `dmkern` instead of function `dmnorm`. 

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

prior = :unif
#prior = :prop
model = kdeda; prior)
fit!(model, Xtrain, ytrain)
pnames(model)
pnames(model.fitm)
fitm = model.fitm ;
fitm.lev
fitm.ni

res = predict(model, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

model = kdeda; prior, a = .5) 
#model = kdeda; prior, h = .1) 
fit!(model, Xtrain, ytrain)
model.fitm.fitm[1].H
```
""" 
function kdeda(X, y; kwargs...) 
    ## To do: add scaling X (?)
    par = recovkw(ParKdeda, kwargs).par
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
    fitm = list(nlev)
    @inbounds for i in eachindex(lev)
        s = y .== lev[i]
        fitm[i] = dmkern(vrow(X, s); h = par.h, a = par.a)
    end
    Kdeda(fitm, priors, lev, ni, par)
end

function predict(object::Kdeda, X)
    X = ensure_mat(X)
    m = nro(X)
    lev = object.lev
    nlev = length(lev) 
    dens = similar(X, m, nlev)
    @inbounds for i in eachindex(lev)
        dens[:, i] .= vec(predict(object.fitm[i], X).pred)
    end
    A = object.priors' .* dens
    v = sum(A, dims = 2)
    posterior = fscale(A', v)'    # Could be replaced by similar as in fscale! 
    z =  mapslices(argmax, posterior; dims = 2)    # if equal, argmax takes the first
    pred = reshape(recod_indbylev(z, object.lev), m, 1)
    (pred = pred, dens, posterior)
end
    
