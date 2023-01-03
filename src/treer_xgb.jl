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
        max_depth = 6, min_child_weight = 5, lambda = 0, 
        scal = false, kwargs...)
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
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl` (https://xgboost.readthedocs.io/en/latest/parameter.html).

The function builds a single tree using package `XGboost.jl' and library XGBoost.

The sampling of the observations and variables is without replacement.

## References

Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. Classification
And Regression Trees. Chapman & Hall, 1984.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245

XGBoost
Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. 
In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
& XGBoost originates from research project at University of Washington.
https://github.com/dmlc/xgboost

XGBoost.jl
https://github.com/dmlc/XGBoost.jl

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
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
        max_depth = 6, min_child_weight = 5, lambda = 0, 
        scal = false, kwargs...) 
    X = ensure_mat(X)
    y = Float64.(vec(y))
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
    num_round = 1
    fm = xgboost((X, y); num_round = num_round,
        seed = Int64(round(rand(1)[1] * 1e5)),
        booster = :gbtree,
        tree_method = "auto", 
        eta = 1, # learning rate
        subsample = subsample,
        colsample_bytree = colsample_bytree, colsample_bylevel = 1,
        colsample_bynode = colsample_bynode, 
        max_depth = max_depth, min_child_weight = min_child_weight,
        lambda = lambda,
        objective = "reg:squarederror", 
        watchlist = (),
        kwargs...)
    featur = collect(1:p)
    TreerXgb(fm, xscales, featur)
end

""" 
    rfr_xgb(X, y; rep = 50,
        subsample = .7,
        colsample_bytree = 1, colsample_bynode = .33,
        max_depth = 6, min_child_weight = 5, lambda = 0, 
        scal = false, kwargs...)
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
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl` (https://xgboost.readthedocs.io/en/latest/parameter.html).

The function uses package `XGboost.jl' and library XGBoost.

The sampling of the observations and variables is without replacement.

## References
Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. 
https://doi.org/10.1007/BF00058655

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. 
https://doi.org/10.1023/A:1010933404324

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
sélection de variables et applications. PhD Thesis. Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245

XGBoost
Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. 
In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
& XGBoost originates from research project at University of Washington.
https://github.com/dmlc/xgboost

XGBoost.jl
https://github.com/dmlc/XGBoost.jl

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
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
        max_depth = 6, min_child_weight = 5, lambda = 0, 
        scal = false, kwargs...)
    X = ensure_mat(X)
    y = Float64.(vec(y))
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
    fm = xgboost((X, y); num_round = 1,
        seed = Int64(round(rand(1)[1] * 1e5)), 
        booster = :gbtree,
        tree_method = "auto",
        num_parallel_tree = rep,
        eta = 1, # learning rate
        subsample = subsample,
        colsample_bytree = colsample_bytree, colsample_bylevel = 1,
        colsample_bynode = colsample_bynode,
        max_depth = max_depth, min_child_weight = min_child_weight,
        lambda = lambda,
        objective = "reg:squarederror", 
        watchlist = (),
        kwargs...)
    featur = collect(1:p)
    TreerXgb(fm, xscales, featur)
end

""" 
    xgboostr(X, y; rep = 50, eta = .3,
        subsample = .7, colsample_bytree = 1, colsample_bynode = 1/3,
        max_depth = 6, min_child_weight = 5, lambda = 1, 
        scal = false, kwargs...)
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
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl` (https://xgboost.readthedocs.io/en/latest/parameter.html).

The function uses package `XGboost.jl' and library XGBoost.

The sampling of the observations and variables is without replacement.

## References
XGBoost
Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. 
In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
& XGBoost originates from research project at University of Washington.
https://github.com/dmlc/xgboost

XGBoost.jl
https://github.com/dmlc/XGBoost.jl

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
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
        max_depth = 6, min_child_weight = 5, lambda = 1, 
        scal = false, kwargs...)
    X = ensure_mat(X)
    y = Float64.(vec(y))
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
    num_round = rep
    fm = xgboost((X, y); num_round = num_round,
        seed = Int64(round(rand(1)[1] * 1e5)), 
        booster = :gbtree,
        tree_method = "auto",
        num_parallel_tree = 1,
        eta = eta,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        colsample_bylevel = 1, colsample_bynode = colsample_bynode,
        max_depth = max_depth, min_child_weight = min_child_weight,
        lambda = lambda, 
        objective = "reg:squarederror",
        watchlist = (), 
        kwargs...)
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

The function uses package `XGboost.jl' and library XGBoost.

## References
XGBoost
Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. 
In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
& XGBoost originates from research project at University of Washington.
https://github.com/dmlc/xgboost

XGBoost.jl
https://github.com/dmlc/XGBoost.jl

## Examples
```julia
using JchemoData, JLD2, CairoMakie, StatsBase
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "challenge2021_cal.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
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
    zimp = XGBoost.importancetable(object.fm)
    res = zeros(p, 5)
    feat = zimp.feature
    for i in eachindex(feat) 
        k = feat[i]
        res[k, 1] = zimp.gain[i]
        res[k, 2] = zimp.weight[i]
        res[k, 3] = zimp.cover[i]
        res[k, 4] = zimp.total_gain[i]
        res[k, 5] = zimp.total_cover[i]
    end
    res = hcat(1:p, res) 
    nam = [:featur, :gain, :weight, :cover, :total_gain, :total_cover]
    DataFrame(res, nam)
end




