"""
    lda(; kwargs...)
    lda(; kwargs...)
    lda(X, y; kwargs...)
    lda(X, y, weights::Weight; kwargs...)
Linear discriminant analysis (LDA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).

The low-level function method (i.e. having argument `weights`) requires to set as input a vector of observation 
weights. In that case, argument `prior` has no effect: the class prior probabilities (output `priors`) are always 
computed by summing the observation weights by class.

In the high-level methods (no argument `weights`), argument `prior` defines how are preliminary computed the 
observation weights (see function `mweightcla`) that are then given as input in the hidden low level method.

**Note:** For highly unbalanced classes, it may be recommended to define equal class weights ('prior = :unif'),
and to use a performance score such as `merrp`, instead of `errp`.

## Examples
```julia
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2")
@load db dat
@names dat
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

model = lda()
fit!(model, Xtrain, ytrain)
@names model
fitm = model.fitm ;
@names fitm
typeof(fitm.fitm) 

fitm.lev
fitm.ni
fitm.priors
aggsumv(fitm.weights.w, ytrain)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt
```
""" 
lda(; kwargs...) = JchemoModel(lda, nothing, kwargs)

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
    n, p = size(X)
    res = matW(X, y, weights)
    ni = res.ni
    priors = aggsumv(weights.w, y).val
    lev = res.lev
    nlev = length(lev)
    res.W .*= n / (n - nlev)    # unbiased estimate
    ## End
    ct = similar(X, nlev, p)
    fitm = list(nlev)
    @inbounds for i in eachindex(lev)
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(vrow(X, s), mweight(weights.w[s]))
        fitm[i] = dmnorm(ct[i, :], res.W)
    end
    Lda(fitm, res.W, ct, ni, priors, lev, weights, par)
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
        dens[:, i] .= vec(predict(object.fitm[i], X).pred)
    end
    A = object.priors' .* dens
    v = sum(A, dims = 2)
    posterior = fscale(A', v)'                    # could be replaced by similar as in fscale! 
    z =  mapslices(argmax, posterior; dims = 2)   # if equal, argmax takes the first
    pred = reshape(recod_indbylev(z, lev), m, 1)
    (pred = pred, dens, posterior)
end
    
