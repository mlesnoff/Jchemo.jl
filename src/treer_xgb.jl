struct TreerXgb
    fm
    featur::AbstractVector
end

struct TreedaXgb
    fm
    featur::AbstractVector
    lev::AbstractVector
    ni::AbstractVector
end

""" 
    treer_xgb(X, y;
        subsample = 1, colsample_bytree = 1, colsample_bynode = 1,
        max_depth = 6, min_child_weight = 5,
        lambda = 0, verbose = false, kwargs...)
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
* `verbose` : If true, fitting information are printed.
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl` (https://xgboost.readthedocs.io/en/latest/parameter.html).

The function builds a single tree using package `XGboost.jl'.

The sampling of the observations and variables are without replacement.

## References

XGBoost.jl
https://github.com/dmlc/XGBoost.jl

Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. Classification
And Regression Trees. Chapman & Hall, 1984.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245
""" 
function treer_xgb(X, y;
        subsample = 1, colsample_bytree = 1, colsample_bynode = 1,
        max_depth = 6, min_child_weight = 5,
        lambda = 0, verbose = false, kwargs...) 
    X = ensure_mat(X)
    p = size(X, 2)
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
    TreerXgb(fm, featur)
end

""" 
    rfr_xgb(X, y; rep = 50,
        subsample = .7,
        colsample_bytree = 1, colsample_bynode = 1/3,
        max_depth = 6, min_child_weight = 5,
        lambda = 0, verbose = false, kwargs...)
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
* `verbose` : If true, fitting information are printed.
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl` (https://xgboost.readthedocs.io/en/latest/parameter.html).

The function uses package `XGboost.jl' to build the forest.
See https://xgboost.readthedocs.io/en/latest/tutorials/rf.html.

## References

XGBoost.jl
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
""" 
function rfr_xgb(X, y; rep = 50,
        subsample = .7,
        colsample_bytree = 1, colsample_bynode = 1/3,
        max_depth = 6, min_child_weight = 5,
        lambda = 0, verbose = false, kwargs...)
    X = ensure_mat(X)
    p = size(X, 2)
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
    TreerXgb(fm, featur)
end

""" 
    xgboostr(X, y; rep = 50, eta = .3,
        subsample = .7, colsample_bytree = 1, colsample_bynode = 1/3,
        max_depth = 6, min_child_weight = 5,
        lambda = 1, verbose = false, kwargs...)
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
* `verbose` : If true, fitting information are printed.
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl` (https://xgboost.readthedocs.io/en/latest/parameter.html).

The function uses package `XGboost.jl' to build the forest.
See https://xgboost.readthedocs.io/en/latest/tutorials/rf.html.

## References

XGBoost 
https://xgboost.readthedocs.io/en/latest/index.html

XGBoost.jl
https://github.com/dmlc/XGBoost.jl
""" 
function xgboostr(X, y; rep = 50, eta = .3,
        subsample = .7, colsample_bytree = 1, colsample_bynode = 1/3,
        max_depth = 6, min_child_weight = 5,
        lambda = 1, verbose = false, kwargs...)
    X = ensure_mat(X)
    p = size(X, 2)
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
    TreerXgb(fm, featur)
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
    pred = XGBoost.predict(object.fm, X)
    pred = reshape(pred, m, 1) ;
    (pred = pred,)
end

"""
    vimp_xgb(object::TreerXgb)
Compute variable (feature) importances from an XGBoost model.
* `object` : The fitted model.

Features with imp = 0 are not returned.
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




