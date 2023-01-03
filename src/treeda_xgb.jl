""" 
    treeda_xgb(X, y;
        subsample = 1, colsample_bytree = 1, colsample_bynode = 1,
        max_depth = 6, min_child_weight = 5, lambda = 0, 
        scal = false, kwargs...)
Discrimination tree with XGBoost.
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
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "forages.jld2") 
@load db dat
pnames(dat)

X = dat.X 
Y = dat.Y 
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]

tab(ytrain)
tab(ytest)

fm = treeda_xgb(Xtrain, ytrain; 
    subsample = .7, 
    col_sample_bynode = 2/3,
    max_depth = 200, min_child_weight = 5) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
res.pred
err(res.pred, ytest)
```
""" 
function treeda_xgb(X, y;
        subsample = 1, colsample_bytree = 1, colsample_bynode = 1,
        max_depth = 6, min_child_weight = 5, lambda = 0, 
        scal = false, kwargs...) 
    X = ensure_mat(X)
    y = vec(y)
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
    ztab = tab(y)
    y_num = recodcat2int(y; start = 0)
    num_class = length(ztab.keys)
    num_round = 1
    fm = xgboost((X, y_num); num_round = num_round,
        seed = Int64(round(rand(1)[1] * 1e5)),
        booster = :gbtree,
        tree_method = "auto", 
        eta = 1, # learning rate
        subsample = subsample,
        colsample_bytree = colsample_bytree, colsample_bylevel = 1,
        colsample_bynode = colsample_bynode, 
        max_depth = max_depth, min_child_weight = min_child_weight,
        lambda = lambda,
        objective = "multi:softmax",
        num_class = num_class,
        watchlist = (),
        kwargs...)
    featur = collect(1:p)
    TreedaXgb(fm, xscales, featur, ztab.keys, ztab.vals)
end

""" 
    rfda_xgb(X, y; rep = 50,
        subsample = .7,
        colsample_bytree = 1, colsample_bynode = 1/3,
        max_depth = 6, min_child_weight = 5, lambda = 0, 
        scal = false, kwargs...)
Random forest discrimination.
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
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "forages.jld2") 
@load db dat
pnames(dat)

Xtrain = dat.Xtrain
ytrain = dat.Ytrain.y
Xtest = dat.Xtest
ytest = dat.Ytest.y

tab(ytrain)
tab(ytest)

fm = rfda_xgb(Xtrain, ytrain; rep = 100, 
    subsample = .7, 
    colsample_bytree = 2 / 3, colsample_bynode = 2 / 3,
    max_depth = 6, min_child_weight = 5) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
res.pred
err(res.pred, ytest)
```
""" 
function rfda_xgb(X, y; rep = 50,
        subsample = .7,
        colsample_bytree = 1, colsample_bynode = 1/3,
        max_depth = 6, min_child_weight = 5, lambda = 0, 
        scal = false, kwargs...)
    X = ensure_mat(X)
    y = vec(y)
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
    ztab = tab(y)
    y_num = recodcat2int(y; start = 0)
    num_class = length(ztab.keys)
    num_round = 1
    fm = xgboost((X, y_num); num_round = num_round,
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
        objective = "multi:softmax",
        num_class = num_class,
        watchlist = (),
        kwargs...)
    featur = collect(1:p)
    TreedaXgb(fm, xscales, featur, ztab.keys, ztab.vals)
end

""" 
    xgboostda(X, y; rep = 50, eta = .3,
        subsample = .7, colsample_bytree = 1, colsample_bynode = 1/3,
        max_depth = 6, min_child_weight = 5, lambda = 1, 
        scal = false, kwargs...)
XGBoost discrimination.
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
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "forages.jld2") 
@load db dat
pnames(dat)

Xtrain = dat.Xtrain
ytrain = dat.Ytrain.y
Xtest = dat.Xtest
ytest = dat.Ytest.y

tab(ytrain)
tab(ytest)

fm = xgboostda(Xtrain, ytrain; rep = 100, 
    subsample = .7, 
    colsample_bytree = 1 / 3, colsample_bynode = 1 / 3,
    max_depth = 6, min_child_weight = 5,
    lambda = .1) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest) ;
res.pred
err(res.pred, ytest)
```
""" 
function xgboostda(X, y; rep = 50, eta = .3,
        subsample = .7, colsample_bytree = 1, colsample_bynode = 1/3,
        max_depth = 6, min_child_weight = 5, lambda = 1, 
        scal = false, kwargs...)
    X = ensure_mat(X)
    y = vec(y)
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
    ztab = tab(y)
    y_num = recodcat2int(y; start = 0)
    num_class = length(ztab.keys)
    num_round = rep
    fm = xgboost((X, y_num); num_round = num_round,
        seed = Int64(round(rand(1)[1] * 1e5)), 
        booster = :gbtree,
        tree_method = "auto",
        num_parallel_tree = 1,
        eta = eta,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        colsample_bylevel = 1, colsample_bynode = colsample_bynode,
        max_depth = max_depth, min_child_weight = min_child_weight,
        objective = "multi:softmax",
        num_class = num_class,
        lambda = lambda, 
        watchlist = (), 
        kwargs...)
    featur = collect(1:p)
    TreedaXgb(fm, xscales, featur, ztab.keys, ztab.vals)
end

"""
    predict(object::TreedaXgb, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::TreedaXgb, X)
    X = ensure_mat(X)
    m = nro(X)
    pred = XGBoost.predict(object.fm, scale(X, object.xscales)) .+ 1
    pred = replacebylev2(Int64.(pred), object.lev)
    pred = reshape(pred, m, 1) ;
    (pred = pred,)
end

