"""
    rda(X, y; gamma, lb, prior = "unif", scal = false)
Regularized discriminant analysis  (QDA).
* `X` : X-data.
* `y` : y-data (class membership).
* `gamma` : Parameyter of shrinkage of the separate covariances of 
    QDA toward a common covariance as in LDA. Must be in [0, 1].
* `lb` : Ridge regularization parameter "lambda" (>= 0).
* `prior` : Type of prior probabilities for class membership.
    Posible values are: "unif" (uniform), "prop" (proportional).

Regularized compromise between LDA and QDA, see Friedman 1989. 

Noting W the pooled within-class (corrected) covariance matrix and 
Wi the within-class (corrected) covariance matrix of class i, the 
regularization is done by with the two successive steps:
* Wi_1 = (1 - `gamma`) * Wi + `gamma` * W
* Wi_2 = Wi_1 + `lb` * I 
Then a QDA is done using matrices Wi_2.

The present function `rda` shrinks the covariance matrices Wi_2 
to the diagonal of the Idendity matrix (ridge regularization;
e.g. Guo et al. 2007), which is slightly different from the 
regularization expression presented by Friedman 1989. Note also 
that parameter `gamma` is referredin Friedman 1989 to as lambda . 
    
## References
Friedman JH. Regularized Discriminant Analysis. Journal of the American 
Statistical Association. 1989; 84(405):165-175. 
doi:10.1080/01621459.1989.10478752.

Guo Y, Hastie T, Tibshirani R. Regularized linear discriminant 
analysis and its application in microarrays. Biostatistics. 
2007; 8(1):86-100. doi:10.1093/biostatistics/kxj035.

## Examples
```julia
using JchemoData, JLD2, StatsBase

path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2") 
@load db dat
pnames(dat)
  
X = dat.X 
Y = dat.Y
y = Y.typ
wl = names(X)
wl_num = parse.(Float64, wl)
ntot = nro(X)

plotsp(X, wl_num).f

summ(Y)
tab(y)
tab(Y.test)

s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(y, s)
Xtest = X[s, :]
ytest = y[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = ntot, ntrain, ntest)

gamma = .2
lb = 1e-7
fm = rda(Xtrain, ytrain; gamma = gamma, lb = lb) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest)
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
        @. res.Wi[i] = res.Wi[i] + lb * Id 
        fm[i] = dmnorm(; mu = ct[i, :], S = res.Wi[i]) 
    end
    Qda(fm, res.Wi, ct, wprior, lev, ni)
end


