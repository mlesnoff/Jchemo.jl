struct TreerXgb
    fm
    xscales::Vector{Float64}
    featur::Vector{Int64}
end

struct TreedaXgb
    fm
    xscales::Vector{Float64}
    featur::Vector{Int64}
    lev::AbstractVector
    ni::AbstractVector
end

""" 
    treer_xgb(X, y;
        subsample = 1, colsample_bytree = 1, colsample_bynode = 1,
        max_depth = 6, min_child_weight = 5,
        lambda = 0, scal = false, verbose = false, 
        kwargs...)
Regression tree with XGBoost.
* `X` : X-data (n obs., p variables).
* `y` : Univariate Y-data (n obs.).
* `subsample` : Proportion of rows sampled in `X` 
    for building the tree.
* `colsample_bytree` : Proportion of columns sampled in `X` 
    for building the tree.
* `colsample_bynode` : Proportion of columns sampled at each node
    in the columns selected for the tree.
* `max_depth` : Maximum depth of the tree.
* `min_child_weight` : Minimum nb. observations that each leaf 
    needs to have.
* `lambda` : L2 regularization term on weights. 
    Increasing this value will make model more conservative.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* `verbose` : If true, fitting information are printed.
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl` (https://xgboost.readthedocs.io/en/latest/parameter.html).

The function builds a single tree using package `XGboost.jl'.

The sampling of the observations and variables are without replacement.

## References
Package XGBoost.jl
https://github.com/dmlc/XGBoost.jl

Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. Classification
And Regression Trees. Chapman & Hall, 1984.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

fm = treer_xgb(Xtrain, ytrain;
    subsample = .7, colsample_bytree = .7, max_depth = 20) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
plotxy(vec(res.pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f   
```
""" 
function treer_xgb(X, y;
        subsample = 1, colsample_bytree = 1, colsample_bynode = 1,
        max_depth = 6, min_child_weight = 5,
        lambda = 0, scal = false, verbose = false, kwargs...) 
    X = ensure_mat(X)
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
    num_round = 1
    fm = xgboost(X, num_round; label = Float64.(vec(y)),
        seed = Int64(round(rand(1)[1] * 1e5)),
        booster = :gbtree,
        tree_method = :auto, 
        eta = 1, # learning rate
        subsample = subsample,
        colsample_bytree = colsample_bytree, colsample_bylevel = 1,
        colsample_bynode = colsample_bynode, 
        max_depth = max_depth, min_child_weight = min_child_weight,
        lambda = lambda,
        silent = !verbose, kwargs...)
    featur = collect(1:p)
    TreerXgb(fm, xscales, featur)
end

""" 
    rfr_xgb(X, y; rep = 50,
        subsample = .7,
        colsample_bytree = 1, colsample_bynode = .33,
        max_depth = 6, min_child_weight = 5,
        lambda = 0, scal = false, verbose = false, kwargs...)
Random forest regression with XGBoost.
* `X` : X-data (n obs., p variables).
* `y` : Univariate Y-data (n obs.).
* `rep` : Nb. trees to build in the forest.
* `subsample` : Proportion of rows sampled in `X` 
    for building each tree.
* `colsample_bytree` : Proportion of columns sampled in `X` at each tree.
* `colsample_bynode` : Proportion of columns sampled at each node 
    in the columns selected for the tree.
* `max_depth` : Maximum depth of the trees.
* `min_child_weight` : Minimum nb. observations that each leaf 
    needs to have.
* `lambda` : L2 regularization term on weights. 
    Increasing this value will make model more conservative.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* `verbose` : If true, fitting information are printed.
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl` (https://xgboost.readthedocs.io/en/latest/parameter.html).

The function uses package `XGboost.jl' to build the forest.
See https://xgboost.readthedocs.io/en/latest/tutorials/rf.html.

## References
Package XGBoost.jl
https://github.com/dmlc/XGBoost.jl

Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. 
https://doi.org/10.1007/BF00058655

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. 
https://doi.org/10.1023/A:1010933404324

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
sélection de variables et applications. PhD Thesis. Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

fm = rfr_xgb(Xtrain, ytrain; rep = 100,
    subsample = .7, colsample_bytree = .7)
pnames(fm)

res = Jchemo.predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
plotxy(vec(res.pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f   
```
""" 
function rfr_xgb(X, y; rep = 50,
        subsample = .7,
        colsample_bytree = 1, colsample_bynode = .33,
        max_depth = 6, min_child_weight = 5,
        lambda = 0, scal = false, verbose = false, kwargs...)
    X = ensure_mat(X)
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
    num_round = 1
    fm = xgboost(X, num_round; label = Float64.(vec(y)),
        seed = Int64(round(rand(1)[1] * 1e5)), 
        booster = :gbtree,
        tree_method = :auto,
        num_parallel_tree = rep,
        eta = 1, # learning rate
        subsample = subsample,
        colsample_bytree = colsample_bytree, colsample_bylevel = 1,
        colsample_bynode = colsample_bynode,
        max_depth = max_depth, min_child_weight = min_child_weight,
        lambda = lambda,
        silent = !verbose, kwargs...)
    featur = collect(1:p)
    TreerXgb(fm, xscales, featur)
end

""" 
    xgboostr(X, y; rep = 50, eta = .3,
        subsample = .7, colsample_bytree = 1, colsample_bynode = 1/3,
        max_depth = 6, min_child_weight = 5,
        lambda = 1, scal = false, verbose = false, kwargs...)
XGBoost regression.
* `X` : X-data (n obs., p variables).
* `y` : Univariate Y-data (n obs.).
* `rep` : Nb. trees to build.
* `eta` : Learning rate ([0, 1]).
* `subsample` : Proportion of rows sampled in `X` 
    for building each tree.
* `colsample_bytree` : Proportion of columns sampled in `X` at each tree.
* `colsample_bynode` : Proportion of columns sampled at each node 
    in the columns selected for the tree.
* `max_depth` : Maximum depth of the trees.
* `min_child_weight` : Minimum nb. observations that each leaf 
    needs to have.
* `lambda` : L2 regularization term on weights. 
    Increasing this value will make model more conservative.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* `verbose` : If true, fitting information are printed.
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl` (https://xgboost.readthedocs.io/en/latest/parameter.html).

The function uses package `XGboost.jl' to build the forest.
See https://xgboost.readthedocs.io/en/latest/tutorials/rf.html.

## References
XGBoost 
https://xgboost.readthedocs.io/en/latest/index.html

Package XGBoost.jl
https://github.com/dmlc/XGBoost.jl

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

fm = xgboostr(Xtrain, ytrain; eta = .1,
    subsample = .7, colsample_bytree = .7)
pnames(fm)

res = Jchemo.predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
plotxy(vec(res.pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f   
```
""" 
function xgboostr(X, y; rep = 50, eta = .3,
        subsample = .7, colsample_bytree = 1, colsample_bynode = 1/3,
        max_depth = 6, min_child_weight = 5,
        lambda = 1, scal = false, verbose = false, kwargs...)
    X = ensure_mat(X)
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
    num_round = rep
    fm = xgboost(X, num_round; label = Float64.(vec(y)),
        seed = Int64(round(rand(1)[1] * 1e5)), 
        booster = :gbtree,
        tree_method = :auto,
        num_parallel_tree = 1,
        eta = eta,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        colsample_bylevel = 1, colsample_bynode = colsample_bynode,
        max_depth = max_depth, min_child_weight = min_child_weight,
        lambda = lambda, 
        silent = !verbose, kwargs...)
    featur = collect(1:p)
    TreerXgb(fm, xscales, featur)
end

"""
    predict(object::TreerXgb, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::TreerXgb, X)
    X = ensure_mat(X)
    m = size(X, 1)
    pred = XGBoost.predict(object.fm, scale(X, object.xscales))
    pred = reshape(pred, m, 1) ;
    (pred = pred,)
end

"""
    vimp_xgb(object::Union{TreerXgb, TreedaXgb})
Compute variable (feature) importances from an XGBoost model.
* `object` : The fitted model.

Features with imp = 0 are not returned.

## Examples
```julia
using JchemoData, JLD2, CairoMakie, StatsBase
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "challenge2021_cal.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
n = nro(X)
wl = names(X)
wl_num = parse.(Float64, wl)

zX = -log.(10, Matrix(X)) 
f = 21 ; pol = 3 ; d = 2 ;
Xp = savgol(snv(X); f, pol, d) ;

fm = xgboostr(Xp, y; rep = 100, 
    subsample = .7, col_sample_bynode = .3,
    max_depth = 6, min_child_weight = 5,
    eta = .1, lambda = 1) ;

res = vimp_xgb(fm)
f, ax = scatter(res.featur, res.gain)
f
```
"""
function vimp_xgb(object::Union{TreerXgb, TreedaXgb})
    p = length(object.featur)
    res = XGBoost.importance(object.fm, string.(object.featur))
    fname = [] ; gain = [] ; cover = [] ; freq = [] 
    for i in res
        push!(fname, i.fname)
        push!(gain, i.gain)
        push!(cover, i.cover)
        push!(freq, i.freq)
    end
    fname = eval(Meta.parse.(fname))
    zgain = zeros(p)
    zcover = copy(zgain)
    zfreq = copy(zgain)
    for i = 1:length(fname)
        s = fname[i]
        zgain[s] = gain[i]
        zcover[s] = cover[i]
        zfreq[s] = freq[i]
    end
    res = DataFrame(hcat(zgain, zcover, zfreq), [:gain, :cover, :freq]) 
    res.featur = object.featur
    res
end




