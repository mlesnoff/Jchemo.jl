struct Lda
    W::Array{Float64}  
    ct::Array{Float64}
    wprior::AbstractVector
    lev::AbstractVector
    ni::AbstractVector
end

"""
    lda(X, y; prior = "unif")
Linear discriminant analysis  (LDA).
* `X` : X-data.
* `y` : y-data (class membership).
* `prior` : Type of prior probabilities for class membership
    ("unif": uniform; "prop": proportional).

## Examples
```julia
using JchemoData, JLD2, StatsBase
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "iris.jld2") 
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
res.ds
res.posterior
res.pred
err(res.pred, ytest)
```
""" 
function lda(X, y; prior = "unif")
    X = ensure_mat(X)
    n = size(X, 1)
    z = aggstat(X; group = y, fun = mean)
    ct = z.X
    lev = z.lev
    nlev = length(lev)
    ni = z.ni
    if isequal(prior, "unif")
        wprior = ones(nlev) / nlev
    elseif isequal(prior, "prop")
        wprior = mweight(ni)
    end
    res = matW(X, y)
    res.W .= res.W * n / (n - nlev) # Unbiased estimate
    Lda(res.W, ct, wprior, lev, ni)
end

"""
    predict(object::Lda, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lda, X)
    X = ensure_mat(X)
    m = size(X, 1)
    lev = object.lev
    nlev = length(lev) 
    ds = similar(X, m, nlev)
    for i = 1:nlev
        fm = dmnorm(; mu = object.ct[i, :], S = object.W) 
        ds[:, i] .= vec(predict(fm, X).pred)
    end
    A = object.wprior' .* ds
    v = sum(A, dims = 2)
    posterior = scale(A', v)' # This could be replaced by code similar as in scale! 
    z =  mapslices(argmax, posterior; dims = 2)  # if equal, argmax takes the first
    pred = reshape(replacebylev2(z, object.lev), m, 1)
    (pred = pred, ds = ds, posterior = posterior)
end
    


