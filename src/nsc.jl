struct Nsc
    ds::Array{Float64}
    cts::Array{Float64}
    d::Array{Float64}
    ct::Array{Float64}
    sel::Vector{Int64}
    poolstd::Vector{Float64}
    s0::Real
    mi::Vector{Float64}
    ni::Vector{Int64}
    lev::AbstractVector
    theta::Vector{Float64}
    delta::Real
    xscales::Vector{Float64}
    weights::Vector{Float64}
end

"""
    nsc(X, y, weights = ones(nro(X)); 
        delta = .5, scal::Bool = false)
Nearest shrunken centroids.
* `X` : X-data.
* `y` : y-data (class membership).
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

function nsc(X, y, weights = ones(nro(X)); 
        delta = .5, scal::Bool = false)
    X = ensure_mat(X)
    y = vec(y)    # for findall
    n, p = size(X)
    weights = mweight(weights)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X, weights)
        X = scale(X, xscales)
    end
    xmeans = colmean(X, weights)
    taby = tab(y)
    lev = taby.keys
    ni = taby.vals
    nlev = length(lev)
    theta = vec(aggstat(weights, y; fun = sum).X)
    ct = similar(X, nlev, p)
    d = copy(ct)
    poolstd = zeros(p)
    mi = similar(X, nlev)
    @inbounds for i = 1:nlev
        s = y .== lev[i]
        ct[i, :] .= colmean(X[s, :], weights[s])
        poolstd .= poolstd .+ theta[i] .* colvar(X[s, :], weights[s])
        d[i, :] .= ct[i, :] .- xmeans
        mi[i] = sqrt(1 / ni[i] - 1 / n)
    end
    poolstd .= sqrt.(poolstd * n / (n - nlev))   # Pooled within-class stds
    s0 = median(poolstd)
    poolstd_s0 = poolstd .+ s0
    scale!(d, poolstd_s0)
    d ./= mi
    ds = soft.(d, delta)
    cts = scale(ds, 1 ./ poolstd_s0) .* mi
    sel = findall(colsum(abs.(ds)) .> 0)
    Nsc(ds, cts, d, ct, sel, poolstd, s0, mi,
        ni, lev, theta, delta, xscales, weights)
end
