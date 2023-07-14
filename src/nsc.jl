struct Nsc5
    ds::Array{Float64}
    cts::Array{Float64}
    d::Array{Float64}
    ct::Array{Float64}
    sel::Vector{Int64}
    selc::AbstractVector
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
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `delta` : Threshold value (>= 0) for the soft thresholding 
    function `soft`.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Compute the nearest shrunken centroids (NSC) proposed by Tibshirani 
et al. (2002). A soft thresholding is used to shrink the class centroids 
and to select the important variables (`X`-columns).  

## References
Tibshirani, R., Hastie, T., Narasimhan, B., Chu, G., 2002. Diagnosis of multiple 
cancer types by shrunken centroids of gene expression. Proceedings of the National 
Academy of Sciences 99, 6567–6572. https://doi.org/10.1073/pnas.082099299

Tibshirani, R., Hastie, T., Narasimhan, B., Chu, G., 2003. Class Prediction 
by Nearest Shrunken Centroids, with Applications to DNA Microarrays. 
Statistical Science 18, 104–117.

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
    cts = xmeans' .+ scale(ds, 1 ./ poolstd_s0) .* mi
    abs_ds = abs.(ds)
    sel = findall(colsum(abs_ds) .> 0)
    selc = list(5, Vector{Int64})
    @inbounds for i = 1:nlev
        selc[i] = findall(abs_ds[i, :] .> 0)
    end 
    Nsc5(ds, cts, d, ct, sel, selc, poolstd, s0, mi,
        ni, lev, theta, delta, xscales, weights)
end
