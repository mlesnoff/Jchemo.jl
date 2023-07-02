struct Qda
    fm
    Wi::AbstractVector  
    ct::Array{Float64}
    wprior::Vector{Float64}
    theta::Vector{Float64}
    ni::Vector{Int64}
    lev::AbstractVector
    weights::Vector{Float64}
end

"""
    qda(X, y, weights = ones(nro(X)); 
        alpha = 0, prior = "unif")
Continuum quadratic discriminant analysis (QDA).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`; default) and LDA (`alpha = 1`).
* `prior` : Type of prior probabilities for class membership.
    Posible values are: "unif" (uniform), "prop" (proportional).

A value `alpha` > 0 shrinks the QDA separate covariances by class 
(Wi) toward a common LDA covariance (W). This corresponds to the first
regularization (Eqs.16) proposed by Friedman 1989.

## References
Friedman JH. Regularized Discriminant Analysis. Journal of the American 
Statistical Association. 1989; 84(405):165-175. 
doi:10.1080/01621459.1989.10478752.

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

prior = "unif"
#prior = "prop"
fm = qda(Xtrain, ytrain; prior = prior) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.dens
res.posterior
res.pred
err(res.pred, ytest)
confusion(res.pred, ytest).cnt
```
""" 
function qda(X, y, weights = ones(nro(X)); 
        alpha = 0, prior = "unif")
    @assert alpha >= 0 && alpha <= 1 "alpha must ∈ [0, 1]"
    # Scaling X has no effect
    X = ensure_mat(X)
    n, p = size(X)
    weights = mweight(weights)
    res = matW(X, y, weights)
    theta = res.theta
    ni = res.ni
    lev = res.lev
    nlev = length(lev)
    res.W .*= n / (n - nlev)    # unbiased estimate
    if isequal(prior, "unif")
        wprior = ones(nlev) / nlev
    elseif isequal(prior, "prop")
        wprior = mweight(ni)
    end
    fm = list(nlev)
    ct = similar(X, nlev, p)
    @inbounds for i = 1:nlev
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(X[s, :], weights[s])
        ni[i] == 1 ? zn = n : zn = ni[i]
        res.Wi[i] .*= zn / (zn - 1)
        if alpha > 0
            @. res.Wi[i] = (1 - alpha) * res.Wi[i] + alpha * res.W
        end
        fm[i] = dmnorm(; mu = ct[i, :], S = res.Wi[i]) 
    end
    Qda(fm, res.Wi, ct, wprior, theta, ni, lev, 
        weights)
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
    A = object.wprior' .* dens
    v = sum(A, dims = 2)
    posterior = scale(A', v)'                    # This could be replaced by code similar as in scale! 
    z =  mapslices(argmax, posterior; dims = 2)  # if equal, argmax takes the first
    pred = reshape(replacebylev2(z, object.lev), m, 1)
    (pred = pred, dens, posterior)
end
    
