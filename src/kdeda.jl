"""
    kdeda(X, y; prior = :unif, h = nothing, a = 1)
Discriminant analysis using non-parametric kernel Gaussian 
    density estimation (KDE-DA).
* `X` : X-data.
* `y` : y-data (class membership).
* `prior` : Type of prior probabilities for class membership.
    Possible values are: :unif (uniform; default), :prop (proportional).
* `h` : See `?dmkern`.
* `h` : See `?dmkern`.

The principle is the same as functions `lda` and `qda` except 
that densities are estimated from `dmkern` instead of  `dmnorm`. 

## Examples
```julia
using JchemoData, JLD2, StatsBase
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

prior = :unif
#prior = :prop
fm = kdeda(Xtrain, ytrain; prior = prior) ;
#fm = kdeda(Xtrain, ytrain; prior = prior, a = .5) ;
#fm = kdeda(Xtrain, ytrain; prior = prior, h = .1) ;
pnames(fm)
fm.fm[1].H

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.dens
res.posterior
res.pred
err(res.pred, ytest)
confusion(res.pred, ytest).cnt
```
""" 
function kdeda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    kdeda(X, y, weights; 
        kwargs...)
end

function kdeda(X, y, weights::Weight; 
        kwargs...) 
    par = recovkwargs(Par, kwargs)
    @assert in([:unif; :prop])(par.prior) "Wrong value for argument 'prior'."
    X = ensure_mat(X)
    Q = eltype(X)
    lev = mlev(y)
    nlev = length(lev)
    ni = tab(y).vals
    if isequal(par.prior, :unif)
        wprior = ones(Q, nlev) / nlev
    elseif isequal(par.prior, :prop)
        wprior = convert.(Q, mweight(ni).w)
    end
    fm = list(nlev)
    for i = 1:nlev
        s = y .== lev[i]
        fm[i] = dmkern(vrow(X, s); 
            h = par.h_kde, a = par.a_kde)
    end
    Kdeda(fm, wprior, lev, ni)
end

function predict(object::Kdeda, X)
    X = ensure_mat(X)
    m = nro(X)
    lev = object.lev
    nlev = length(lev) 
    dens = similar(X, m, nlev)
    ni = object.ni
    for i = 1:nlev
        dens[:, i] .= vec(Jchemo.predict(object.fm[i], X).pred)
    end
    A = object.wprior' .* dens
    v = sum(A, dims = 2)
    posterior = fscale(A', v)'                    # This could be replaced by code similar as in fscale! 
    z =  mapslices(argmax, posterior; dims = 2)  # if equal, argmax takes the first
    pred = reshape(replacebylev2(z, object.lev), m, 1)
    (pred = pred, dens = dens, posterior = posterior)
end
    
