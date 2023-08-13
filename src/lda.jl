struct Lda
    fm
    W::Array{Float64}  
    ct::Array{Float64}
    wprior::Vector{Float64}
    theta::Vector{Float64}
    ni::Vector{Int64}
    lev::AbstractVector
    weights::Vector{Float64}
end

"""
    lda(X, y, weights = ones(nro(X)); 
        prior = "unif")
Linear discriminant analysis  (LDA).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `prior` : Type of prior probabilities for class membership.
    Possible values are: "unif" (uniform), "prop" (proportional).

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
fm = lda(Xtrain, ytrain; prior = prior) ;
pnames(fm)
println(typeof(fm))

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.dens
res.posterior
res.pred
err(res.pred, ytest)
confusion(res.pred, ytest).cnt
```
""" 
function lda(X, y, weights = ones(nro(X)); 
        prior = "unif")
    @assert in(["unif"; "prop"])(prior) "Wrong value for argument 'prior'."
    # Scaling X has no effect
    X = ensure_mat(X)
    y = vec(y)    # for findall
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
        fm[i] = dmnorm(; mu = ct[i, :], S = res.W) 
    end
    Lda(fm, res.W, ct, wprior, theta, ni, lev, 
        weights)
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
        dens[:, i] .= vec(Jchemo.predict(object.fm[i], X).pred)
    end
    A = object.wprior' .* dens
    v = sum(A, dims = 2)
    posterior = scale(A', v)'                    # This could be replaced by code similar as in scale! 
    z =  mapslices(argmax, posterior; dims = 2)  # if equal, argmax takes the first
    pred = reshape(replacebylev2(z, lev), m, 1)
    (pred = pred, dens, posterior)
end
    
