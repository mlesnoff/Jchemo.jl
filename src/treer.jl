struct Treer
    fm
end

""" 
    treer_dt(X, y; samp_col_node = 1, max_depth = 6, 
        min_leaf = 5, kwargs...)
Tree model (CART) for regression.
* `X` : X-data.
* `y` : Y-data (univariate).
* `samp_col_node` : Proportion of columns (variables) sampled in `X` at each node.
* `max_depth` : Maximum depth of the tree.
* `min_leaf` :  Minimum nb. observations that each leaf 
    needs to have.
* `min_split`: Minimum nb. observations needed for a split.
* `kwargs` : Optional named arguments to pass in function `build_tree` 
    of `DecisionTree.jl`.

The function uses package `DecisionTree.jl' to build the tree.

## References

DecisionTree.jl
https://github.com/bensadeghi/DecisionTree.jl

Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. Classication
And Regression Trees. Chapman & Hall, 1984.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245
""" 
function treer_dt(X, y; samp_col_node = 1, max_depth = 6, 
    min_leaf = 5, kwargs...) 
    X = ensure_mat(X)
    y = vec(y)
    p = size(X, 2)
    n_col_node = max(Int64(round(samp_col_node * p)), 1)
    min_split = 2
    min_purity_increase = 0
    fm = build_tree(y, X,
        n_col_node,
        max_depth, min_leaf, min_split,
        min_purity_increase)
    Treer(fm)
end

""" 
    treer_xgb(X, y; samp_col_node = 1, max_depth = 6, 
        min_leaf = 5, kwargs...)
Tree model (CART) for regression.
* `X` : X-data.
* `y` : Y-data (univariate).
* `samp_col_node` : Proportion of columns (variables) sampled in `X` at each node.
* `max_depth` : Maximum depth of the tree.
* `min_leaf` : Minimum nb. observations that each leaf 
    needs to have.
* `min_split`: Minimum nb. observations needed for a split.
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl`.

The function uses package `XGboost.jl' to build the tree.

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
        max_depth = nothing, colsample_bynode = 1, 
        min_child_weight = 5, 
        tree_method = :auto, kwargs...) 
    X = ensure_mat(X)
    y = Float64.(vec(y))
    isnothing(max_depth) ? max_depth = length(y) : nothing
    num_round = 1
    fm = xgboost(X, num_round; label = y,
        #seed = Int64(round(rand(1)[1] * 10000)), 
        eta = 1,
        lambda = 0,
        subsample = 1,
        colsample_bytree = 1,
        colsample_bylevel = 1,
        colsample_bynode = colsample_bynode, 
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        tree_method = tree_method,
        verbosity = 0, kwargs...)
    Treer(fm)
end

""" 
    treer_evt(X, y; samp_col_node = 1, max_depth = 6, 
        min_leaf = 5, nbins = 64)
Tree model (CART) for regression.
* `X` : X-data.
* `y` : Y-data (univariate).
* `samp_col_node` : Proportion of columns (variables) sampled in `X` at each node.
* `max_depth` : Maximum depth of the tree.
* `min_leaf` : Minimum nb. observations that each leaf 
    needs to have.
* `min_split`: Minimum nb. observations needed for a split.
* `nbins` : Nb. bins in the histogram computations.

The function uses package `EvoTrees.jl' to build the tree.

## References

EvoTrees.jl
https://github.com/Evovest/EvoTrees.jl

Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. Classication
And Regression Trees. Chapman & Hall, 1984.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245
""" 
function treer_evt(X, y; 
        max_depth = 6, min_weight = 5, nbins = 64) 
    X = ensure_mat(X)
    y = vec(y)
    isequal(max_depth, -1) ? max_depth = Int64(1e6) : nothing
    nbins = min(nbins, 255)
    params1 = EvoTreeRegressor(
        loss = :linear,
        nrounds = 1,
        η = 1,
        rowsample = 1.0, 
        colsample = 1.0, 
        max_depth = max_depth, 
        min_weight = Float64(min_weight),
        λ = 0, γ = 0,
        nbins = nbins)
    fm = fit_evotree(params1, X, y)
    Treer(fm)
end

function predict(object::Treer, X)
    X = ensure_mat(X)
    m = size(X, 1)
    if in(propertynames(object.fm)).(:featid)   # DecisionTree
        pred = apply_tree(object.fm, X)
    elseif isa(object.fm, Booster)   # XGBoost
        pred = XGBoost.predict(object.fm, X)
    elseif isa(object.fm, EvoTrees.GBTree{Float64})  # EvoTrees
        pred = EvoTrees.predict(object.fm, X)
    end
    pred = reshape(pred, m, 1) ;
    (pred = pred,)
end

