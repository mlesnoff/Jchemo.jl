struct Xgboost
    fm
end

""" 
    xgboostr(X, y; booster = "gbtree", obj = "reg:squarederror",
        B, eta = 1, lambda = 0, alpha = 0, 
        samp_row = 1, samp_col = 1, samp_col_node = 1, 
        max_depth = 6, min_leaf = 5, kwargs...) 
XGBoost for regression.
* `X` : X-data.
* `y` : Y-data (univariate).
* `B` (num_round) : Nb. of boosting iterations.
* `obj` (objective) : Lerning task and objective.
* `booster` : Type of booster ("gbtree", "gblinear" or "dart").
* `eta` : Step size shrinkage used in update to prevents overfitting
    (must be between 0 and 1; default: 1).
* `lambda` : L2 regularization term on weights. 
    Increasing this value will make model more conservative.
* `alpha` : L1 regularization term on weights.
    Increasing this value will make model more conservative.
* `samp_row` (subsample) : Proportion of rows (observations)
    sampled in `X` at each tree.
* `samp_col` (colsample_bytree) : Proportion of columns (variables) 
    sampled in `X` at each tree.
* `samp_col_node` (colsample_bynode): Proportion of columns (variables)
    sampled in `X` at each node.
* `max_depth` : Maximum depth of the tree.
* `min_leaf` (min_child_weight) :  Minimum nb. observations that each leaf 
    needs to have.
* `kwargs` : Optional named arguments to pass in function `xgboost` 
    of `XGBoost.jl`.

The function uses package `XGBoost.jl'. 

XGBoost.jl uses the library XGBoost (https://xgboost.readthedocs.io/en/latest/index.html).

## References

XGBoost.jl
https://github.com/dmlc/XGBoost.jl

""" 
function xgboostr(X, y; booster = "gbtree", obj = "reg:squarederror",
    B, eta = 1, lambda = 0, alpha = 0, 
    samp_row = 1, samp_col = 1, samp_col_node = 1, 
    max_depth = 6, min_leaf = 5, kwargs...) 
    X = ensure_mat(X)
    y = vec(y)
    fm = xgboost(X, B; label = y, 
        booster = Symbol(booster), objective = obj,
        eta = eta, lambda = lambda, alpha = alpha, 
        subsample = samp_row,
        colsample_bytree = samp_col, colsample_bynode = samp_col_node,
        min_child_weight = min_leaf, max_depth = max_depth, 
        verbosity = 0, kwargs...) ;
    Xgboost(fm)
end

function predict(object::Xgboost, X)
    X = ensure_mat(X)
    m = size(X, 1)
    pred = XGBoost.predict(object.fm, X)
    pred = reshape(pred, m, 1) ;
    (pred = pred,)
end

