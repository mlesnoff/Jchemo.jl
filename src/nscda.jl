struct Nscda4
    fms
    poolstd_s0::Vector{Float64}
    wprior::Vector{Float64}
    ni::Vector{Int64}
    lev::AbstractVector
    xscales::Vector{Float64}
    weights::Vector{Float64}
end

"""
    nscda(X, y, weights = ones(nro(X)); delta = .5, 
        prior = "unif", scal::Bool = false)
Discrimination by nearest shrunken centroids (NSC).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `delta` : Threshold value (>= 0) for function `soft`
    (soft thresholding). 
* `prior` : Type of prior probabilities for class membership.
    Posible values are: "unif" (uniform), "prop" (proportional).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

The new observations to predict are classified by computing the distnace to the 
shrunken class centroids (Tibshirani et al. 2002, 2003). 

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
prior = "prop"
fm = nscda(Xtrain, ytrain; delta = delta, 
    prior = prior) ;
res = Jchemo.predict_nscda(fm, Xtest) ; 
res.d2
err(res.pred, ytest)
```
""" 
function nscda(X, y, weights = ones(nro(X)); delta, 
        prior = "unif", scal::Bool = false)
    weights = mweight(weights)
    fms = nsc(X, y, weights;
        delta = delta, scal = scal)
    poolstd_s0 = fms.poolstd .+ fms.s0
    nlev = length(fms.lev)
    if isequal(prior, "unif")
        wprior = ones(nlev) / nlev
    elseif isequal(prior, "prop")
        wprior = mweight(fms.ni)
    end
    Nscda4(fms, poolstd_s0, wprior, fms.ni, 
        fms.lev, fms.xscales, weights)
end

"""
    predict(object::Dmnorm, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : Data (vector) for which predictions are computed.
""" 
function predict(object::Nscda4, X)
    zX = scale(X, object.xscales)
    m = nro(zX)
    scale!(zX, object.poolstd_s0)
    cts = scale(object.fms.cts, object.poolstd_s0)
    d2 = euclsq(zX, cts) .- 2 * log.(object.wprior)'
    posterior = softmax(-.5 * d2)
    z =  mapslices(argmin, d2; dims = 2)  # if equal, argmin takes the first
    pred = reshape(replacebylev2(z, object.lev), m, 1)
    (pred = pred, d2, posterior)
end

