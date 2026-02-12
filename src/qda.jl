"""
    qda(; kwargs...)
    qda(X, y; kwargs...)
    qda(X, y, weights::Weight; kwargs...)
Quadratic discriminant analysis (QDA, with continuum towards LDA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g., function `mweight`).
Keyword arguments:
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum between QDA (`alpha = 0`) and LDA (`alpha = 1`).

The low-level function method (i.e. having argument `weights`) requires to set as input a vector of observation 
weights. In that case, argument `prior` has no effect: the class prior probabilities (output `priors`) are always 
computed by summing the observation weights by class.

In the high-level methods (no argument `weights`), argument `prior` defines how are preliminary computed the 
observation weights (see function `mweightcla`) that are then given as input in the hidden low level method.

**Note:** For highly unbalanced classes, it may be recommended to define equal class weights ('prior = :unif'),
and to use a performance score such as `merrp`, instead of `errp`.

For the continuum approach, a value `alpha` > 0 shrinks the QDA class covariances (Wi) toward a common LDA 
covariance ('within-W'). This corresponds to the 'first regularization (Eqs.16)' approach described in 
Friedman 1989 (in which the present parameter `alpha` is referred to as 'lambda').

## References
Friedman JH. Regularized Discriminant Analysis. Journal of the American Statistical Association. 1989; 84(405):165-175. 
doi:10.1080/01621459.1989.10478752.

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

model = qda()
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

## With regularization
alpha = 0.5
#alpha = 1 # = LDA
model = qda(; alpha)
fit!(model, Xtrain, ytrain)
model.fitm.Wi
res = predict(model, Xtest) ;
errp(res.pred, ytest)
```
""" 
qda(; kwargs...) = JchemoModel(qda, nothing, kwargs)

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
    priors = aggsumv(weights.w, vec(y)).val
    lev = res.lev
    nlev = length(lev)
    res.W .*= n / (n - nlev)    # unbiased estimate
    ## End
    ct = similar(X, nlev, p)
    fitm = list(nlev)
    @inbounds for i in eachindex(lev)
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(vrow(X, s), mweight(weights.w[s]))
        if alpha > 0
            @. res.Wi[i] = (1 - alpha) * res.Wi[i] + alpha * res.W
        end
        fitm[i] = dmnorm(ct[i, :], res.Wi[i]) 
    end
    Qda(fitm, res.Wi, ct, ni, priors, lev, weights, par)
end

