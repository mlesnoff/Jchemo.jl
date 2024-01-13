"""
    rda(; kwargs...)
    rda(X, y; kwargs...)
    rda(X, Y, weights::Weight; kwargs...)
Regularized discriminant analysis (RDA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum
    between QDA (`alpha = 0`) and LDA (`alpha = 1`).
* `lb` : Ridge regularization parameter "lambda" (>= 0).
* `simpl` : Boolean. See function `dmnorm`. 
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Let us note W the (corrected) pooled within-class 
covariance matrix and Wi the (corrected) within-class 
covariance matrix of class i. The regularization is done 
by the two following successive steps (for each class i):
* Continuum between QDA and LDA: Wi(1) = (1 - `alpha`) * Wi + `alpha` * W       
* Ridge regularization: Wi(2) = Wi(1) + `lb` * I
Then the QDA algorithm is run on matrices {Wi(2)}.

Function `rda` is slightly different from the regularization 
expression used by Friedman 1989 (Eq.18). It shrinks the 
covariance matrices Wi(2) to the diagonal of the Idendity 
matrix (ridge regularization; e.g. Guo et al. 2007).  

Particular cases:
* `alpha` = 1 & `lb` = 0 : LDA
* `alpha` = 0 & `lb` = 0 : QDA
* `alpha` = 1 & `lb` > 0 : Penalized LDA 
    (Hastie et al 1995) with diagonal regularization 
    matrix

## References
Friedman JH. Regularized Discriminant Analysis. 
Journal of the American Statistical Association. 1989; 
84(405):165-175. doi:10.1080/01621459.1989.10478752.

Guo Y, Hastie T, Tibshirani R. Regularized linear 
discriminant analysis and its application in microarrays. 
Biostatistics. 2007; 8(1):86-100. 
doi:10.1093/biostatistics/kxj035.

Hastie, T., Buja, A., Tibshirani, R., 1995. Penalized 
Discriminant Analysis. The Annals of Statistics 23, 73–102.

## Examples
```julia
using JchemoData, JLD2
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

alpha = .5
lb = 1e-8
mod = rda(; alpha, lb)
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
fm.lev
fm.ni

res = predict(mod, Xtest) ;
pnames(res)
@head res.posterior
@head res.pred
errp(res.pred, ytest)
confusion(res.pred, ytest).cnt
```
""" 
function rda(X, y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    rda(X, y, weights; 
        kwargs...)
end

function rda(X, Y, weights::Weight; kwargs...)  
    par = recovkwargs(Par, kwargs)
    @assert in([:unif; :prop])(par.prior) "Wrong value for argument 'prior'."
    @assert 0 <= par.alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."
    @assert par.lb >= 0 "lb must be in >= 0"
    X = ensure_mat(X)
    y = vec(y)    # for findall
    Q = eltype(X)
    n, p = size(X)
    alpha = convert(Q, par.alpha)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        X = fscale(X, xscales)
    end
    res = matW(X, y, weights)
    ni = res.ni
    lev = res.lev
    nlev = length(lev)
    if isequal(par.prior, :unif)
        wprior = ones(Q, nlev) / nlev
    elseif isequal(par.prior, :prop)
        wprior = convert.(Q, mweight(ni).w)
    end
    fm = list(nlev)
    ct = similar(X, nlev, p)
    Id = I(p)
    fm = list(nlev)
    res.W .*= n / (n - nlev)    # unbiased estimate
    @inbounds for i = 1:nlev
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(X[s, :], 
            mweight(weights.w[s]))
        ni[i] == 1 ? zn = n : zn = ni[i]
        res.Wi[i] .*= zn / (zn - 1)        
        @. res.Wi[i] = (1 - alpha) * res.Wi[i] + alpha * res.W
        @. res.Wi[i] = res.Wi[i] + par.lb * Id 
        fm[i] = dmnorm(; mu = ct[i, :], 
            S = res.Wi[i], simpl = par.simpl) 
    end
    Rda(fm, res.Wi, ct, wprior, res.theta.w,
        ni, lev, xscales, weights)
end

"""
    predict(object::Qda, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Rda, X)
    X = ensure_mat(X)
    m = nro(X)
    X = fscale(X, object.xscales)
    lev = object.lev
    nlev = length(lev) 
    dens = similar(X, m, nlev)
    for i = 1:nlev
        dens[:, i] .= vec(Jchemo.predict(object.fm[i], 
            X).pred)
    end
    A = object.wprior' .* dens
    v = sum(A, dims = 2)
    posterior = fscale(A', v)'  # Could be replaced by similar as in fscale! 
    z =  mapslices(argmax, posterior; 
        dims = 2)  # if equal, argmax takes the first
    pred = reshape(replacebylev2(z, 
        lev), m, 1)
    (pred = pred, dens, posterior)
end

