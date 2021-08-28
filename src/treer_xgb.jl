struct TreerXgb
    fm
end

""" 
    treer_xgb(X, y; subsample = 1,
        colsample_bytree = 1, colsample_bynode = 1,
        max_depth = 6, min_child_weight = 5,
        lambda = 0, kwargs...)
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
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl` (https://xgboost.readthedocs.io/en/latest/parameter.html).

The function builds a single tree using package `XGboost.jl'.

The sampling of the observations and variables are without replacement.

## References

XGBoost.jl
https://github.com/dmlc/XGBoost.jl

Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. Classication
And Regression Trees. Chapman & Hall, 1984.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245
""" 
function treer_xgb(X, y;
        subsample = 1,
        colsample_bytree = 1, 
        colsample_bynode = 1,
        max_depth = 6, 
        min_child_weight = 5,
        lambda = 0, 
        kwargs...) 
    X = ensure_mat(X)
    y = Float64.(vec(y))
    num_round = 1
    fm = xgboost(X, num_round; label = y,
        seed = Int64(round(rand(1)[1] * 1e5)), 
        eta = 1,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        colsample_bylevel = 1,
        colsample_bynode = colsample_bynode, 
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        lambda = lambda,
        verbosity = 0, kwargs...)
    TreerXgb(fm)
end

""" 
    rfr_xgb(X, y; B, subsample = .7,
        colsample_bytree = 1, colsample_bynode = 1/3, 
        max_depth = 6, min_child_weight = 5, 
        lambda = 0, kwargs...)
Random forest regression with XGBoost.
* `X` : X-data (n obs., p variables).
* `y` : Univariate Y-data (n obs.).
* `B` : Nb. trees to build in the forest.
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
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl` (https://xgboost.readthedocs.io/en/latest/parameter.html).

The function uses package `XGboost.jl' to build the forest.
See https://xgboost.readthedocs.io/en/latest/tutorials/rf.html.

## References

XGBoost.jl
https://github.com/dmlc/XGBoost.jl

Breiman, L. (2001). Random forests. Mach Learn, 45:5-32. 
doi: 10.1023/A:1010933404324.

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, sélection de variables 
et applications (PhD thesis). Université Paris Sud - Paris XI.
""" 
function rfr_xgb(X, y;
        B,
        subsample = .7,
        colsample_bytree = 1,
        colsample_bynode = 1/3,
        max_depth = 6, 
        min_child_weight = 5,
        lambda = 0, 
        kwargs...)
    X = ensure_mat(X)
    y = Float64.(vec(y))
    num_round = 1
    fm = xgboost(X, num_round; label = Float64.(vec(y)),
        seed = Int64(round(rand(1)[1] * 1e5)), 
        booster = :gbtree,
        tree_method = :auto,
        num_parallel_tree = B,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        colsample_bylevel = 1,
        colsample_bynode = colsample_bynode,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        lambda = lambda,
        eta = 1,
        verbosity = 0)
    TreerXgb(fm)
end

""" 
    xgboostr(X, y; B, subsample = .7,
        colsample_bytree = 1, colsample_bynode = 1/3, 
        max_depth = 6, min_child_weight = 5, lambda = .3, kwargs...)
XGBoost regression.
* `X` : X-data (n obs., p variables).
* `y` : Univariate Y-data (n obs.).
* `B` : Nb. trees to build.
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
function xgboostr(X, y;
        B,
        eta = .3,
        subsample = .7,
        colsample_bytree = 1,
        colsample_bynode = 1/3,
        max_depth = 6, 
        min_child_weight = 5,
        lambda = 1, 
        kwargs...)
    X = ensure_mat(X)
    y = Float64.(vec(y))
    num_round = B
    fm = xgboost(X, num_round; label = Float64.(vec(y)),
        seed = Int64(round(rand(1)[1] * 1e5)), 
        booster = :gbtree,
        tree_method = :auto,
        num_parallel_tree = 1,
        eta = eta,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        colsample_bylevel = 1,
        colsample_bynode = colsample_bynode,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        lambda = lambda,
        verbosity = 0, kwargs...)
    TreerXgb(fm)
end

function predict(object::TreerXgb, X)
    X = ensure_mat(X)
    m = size(X, 1)
    pred = XGBoost.predict(object.fm, X)
    pred = reshape(pred, m, 1) ;
    (pred = pred,)
end

