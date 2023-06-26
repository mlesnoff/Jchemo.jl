"""
    rda(X, y; alpha, lb, prior = "unif", scal = false)
Regularized discriminant analysis  (RDA).
* `X` : X-data.
* `y` : y-data (class membership).
* `alpha` : Shrinkage parameter of the separate covariances of 
    QDA toward a common covariance as in LDA. Must be in [0, 1].
* `lb` : Ridge regularization parameter "lambda" (>= 0).
* `prior` : Type of prior probabilities for class membership.
    Posible values are: "unif" (uniform), "prop" (proportional).

Regularized compromise between LDA and QDA, see Friedman 1989. 

Noting W the (corrected) pooled within-class covariance matrix and 
Wi the (corrected) within-class covariance matrix of class i, the 
regularization is done by with the two successive steps:
* Wi(1) = (1 - `alpha`) * Wi + `alpha` * W       (compromise between LDA and QDA)
* Wi(2) = Wi(1) + `lb` * I       (ridge regularization)
Then a QDA is done using matrices Wi_2.

Function `rda` shrinks the covariance matrices Wi(2) 
to the diagonal of the Idendity matrix (ridge regularization)
(e.g. Guo et al. 2007). This is slightly different from the 
regularization expression used by Friedman 1989. 
Note: Parameter `alpha` is referred to as lambda in Friedman 1989.

Particular cases:
* `alpha` = 1 & `lb` = 0 : LDA
* `alpha` = 0 & `lb` = 0 : QDA

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

alpha = .2
lb = 1e-7
fm = rda(Xtrain, ytrain; alpha = alpha, lb = lb) ;
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
function rda(X, y; alpha, lb, prior = "unif")
    @assert alpha >= 0 && alpha <= 1 "alpha must be in [0, 1]"
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
        @. res.Wi[i] = (1 - alpha) * res.Wi[i] + alpha * res.W
        @. res.Wi[i] = res.Wi[i] + lb * Id 
        fm[i] = dmnorm(; mu = ct[i, :], S = res.Wi[i]) 
    end
    Qda(fm, res.Wi, ct, wprior, lev, ni)
end

