struct Rda
    fm
    Wi::AbstractVector  
    ct::Array{Float64}
    wprior::Vector{Float64}
    theta::Vector{Float64}
    ni::Vector{Int64}
    lev::AbstractVector
    xscales::Vector{Float64}
    weights::Vector{Float64}
end

"""
    rda(X, y, weights = ones(nro(X)); 
        prior = "unif", alpha = 1, lb = 1e-10, 
        simpl::Bool = false, scal::Bool = false)
Regularized discriminant analysis (RDA).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `prior` : Type of prior probabilities for class membership.
    Posible values are: "unif" (uniform), "prop" (proportional).
* `alpha` : Shrinkage parameter of the separate covariances of 
    QDA toward a common covariance as in LDA. Must ∈ [0, 1]
    (`alpha` is referred to as lambda in Friedman 1989).
* `lb` : Ridge regularization parameter "lambda" (>= 0).
* `simpl` : Boolean (default to `false`). See `dmnorm`. 
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Regularized compromise between LDA and QDA, see Friedman 1989. 

Noting W the (corrected) pooled within-class covariance matrix and 
Wi the (corrected) within-class covariance matrix of class i, the 
regularization is done by with the two successive steps:
* Continuum between QDA and LDA: Wi(1) = (1 - `alpha`) * Wi + `alpha` * W       
* Ridge regularization: Wi(2) = Wi(1) + `lb` * I
Then the QDA algorithm is run on matrices Wi(2).

Function `rda` is slightly different from the regularization expression 
used by Friedman 1989 (Eq.18). It shrinks the covariance matrices Wi(2) 
to the diagonal of the Idendity matrix (ridge regularization)
(e.g. Guo et al. 2007).  

Particular cases:
* `alpha` = 1 & `lb` = 0 : LDA
* `alpha` = 0 & `lb` = 0 : QDA
* `alpha` = 1 & `lb` > 0 : Penalized LDA (Hstie et al 1995) with diagonal
    regularization matrix

## References
Friedman JH. Regularized Discriminant Analysis. Journal of the American 
Statistical Association. 1989; 84(405):165-175. 
doi:10.1080/01621459.1989.10478752.

Guo Y, Hastie T, Tibshirani R. Regularized linear discriminant 
analysis and its application in microarrays. Biostatistics. 
2007; 8(1):86-100. doi:10.1093/biostatistics/kxj035.

Hastie, T., Buja, A., Tibshirani, R., 1995. Penalized Discriminant Analysis. 
The Annals of Statistics 23, 73–102.

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

alpha = .5
lb = 1e-8
fm = rda(Xtrain, ytrain; alpha = alpha, 
    lb = lb, simpl = true) ;
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
function rda(X, y, weights = ones(nro(X)); 
        prior = "unif", alpha = 1, lb = 1e-10, 
        simpl::Bool = false, scal::Bool = false)
    @assert alpha >= 0 && alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."
    @assert lb >= 0 "lb must be in >= 0"
    X = ensure_mat(X)
    y = vec(y)    # for findall
    n, p = size(X)
    weights = mweight(weights)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X, weights)
        X = scale(X, xscales)
    end
    res = matW(X, y, weights)
    ni = res.ni
    lev = res.lev
    nlev = length(lev)
    if isequal(prior, "unif")
        wprior = ones(nlev) / nlev
    elseif isequal(prior, "prop")
        wprior = mweight(ni)
    end
    fm = list(nlev)
    ct = similar(X, nlev, p)
    Id = I(p)
    fm = list(nlev)
    res.W .*= n / (n - nlev)    # unbiased estimate
    @inbounds for i = 1:nlev
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(X[s, :], weights[s])
        ni[i] == 1 ? zn = n : zn = ni[i]
        res.Wi[i] .*= zn / (zn - 1)        
        @. res.Wi[i] = (1 - alpha) * res.Wi[i] + alpha * res.W
        @. res.Wi[i] = res.Wi[i] + lb * Id 
        fm[i] = dmnorm(; mu = ct[i, :], S = res.Wi[i],
            simpl = simpl) 
    end
    Rda(fm, res.Wi, ct, wprior, res.theta, ni, lev, 
        xscales, weights)
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
    X = scale(X, object.xscales)
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

