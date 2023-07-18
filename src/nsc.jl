"""
    nsc(X, y, weights = ones(nro(X)); 
        delta = .5, scal::Bool = false)
Nearest shrunken centroids (NSC).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `delta` : Threshold value (>= 0) for function `soft`
    (soft thresholding). 
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Compute the nearest shrunken centroids (NSC) proposed by Tibshirani 
et al. (2002, 2003). 

A soft thresholding is used to shrink the class centroids 
and to select the important `X`-variables (columns).  

## References
Tibshirani, R., Hastie, T., Narasimhan, B., Chu, G., 2002. Diagnosis of multiple 
cancer types by shrunken centroids of gene expression. Proceedings of the National 
Academy of Sciences 99, 6567–6572. https://doi.org/10.1073/pnas.082099299

Tibshirani, R., Hastie, T., Narasimhan, B., Chu, G., 2003. Class Prediction 
by Nearest Shrunken Centroids, with Applications to DNA Microarrays. 
Statistical Science 18, 104–117.

## Examples
```julia
using JchemoData, JLD2
using FreqTables

path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/srbct.jld2") 
@load db dat
pnames(dat)

Xtrain = dat.Xtrain
Ytrain = dat.Ytrain
ytrain = Ytrain.y
Xtest = dat.Xtest
Ytest = dat.Ytest
s = Ytest.y .<= 4
Xtest = Xtest[s, :]
Ytest = Ytest[s, :]
ytest = Ytest.y
ntrain, p = size(Xtrain)
ntest = nro(Xtest)
(ntrain = ntrain, ntest)

freqtable(ytrain, Ytrain.lab)

delta = 4.34
fm = nsc(Xtrain, ytrain; delta = delta) ;
pnames(fm)
fm.ct       # centroids
fm.d        # statistic d (eq.[1] in Tibshirani et al. 2002)  
fm.cts      # shrunken centroids (eq.[4] in Tibshirani et al. 2002)
fm.ds       # statistic d' (eq.[5] in Tibshirani et al. 2002)
fm.sel      # indexes of the selected variables
i = 1
fm.selc[i]  # indexes of the selected variables for class i
```
""" 
function nsc(X, y, weights = ones(nro(X)); 
        delta = .5, scal::Bool = false)
    X = ensure_mat(X)
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
    selc = list(nlev, Vector{Int64})
    @inbounds for i = 1:nlev
        selc[i] = findall(abs_ds[i, :] .> 0)
    end 
    (ds = ds, d, cts, ct, sel, selc, poolstd, s0, 
        mi, ni, lev, theta, xscales, weights)
end
