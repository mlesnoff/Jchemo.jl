"""
    rda(X, y; gamma, lb, prior = "unif", scal = false)
Quadratic discriminant analysis  (QDA).
* `X` : X-data.
* `y` : y-data (class membership).
* `gamma` : .
* `lb` : Ridge regularization parameter "lambda".
* `prior` : Type of prior probabilities for class membership.
    Posible values are: "unif" (uniform), "prop" (proportional).

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


function rda(X, y; gamma, lb, prior = "unif")
    @assert gamma >= 0 && gamma <= 1 "gamma must be in [0, 1]"
    @assert lb >= 0 "lb must be in >= 0"
    X = ensure_mat(X)
    n, p = size(X)
    z = aggstat(X, y; fun = mean)
    ct = z.X
    lev = z.lev
    nlev = length(lev)
    res = matW(X, y)
    ni = res.ni
    if isequal(prior, "unif")
        wprior = ones(nlev) / nlev
    elseif isequal(prior, "prop")
        wprior = mweight(ni)
    end
    res.W .*= n / (n - nlev)
    Id = I(p)
    fm = list(nlev)
    @inbounds for i = 1:nlev
        ni[i] == 1 ? zn = n : zn = ni[i]
        res.Wi[i] .*= zn / (zn - 1)        
        @. res.Wi[i] = (1 - gamma) * res.Wi[i] + gamma * res.W
        #res.Wi[i] .= (1 - gamma) .* res.Wi[i] .+ gamma .* res.W
        res.Wi[i] .= res.Wi[i] .+ lb .* Id
        fm[i] = dmnorm(; mu = ct[i, :], S = res.Wi[i]) 
    end
    Qda(fm, res.Wi, ct, wprior, lev, ni)
end


