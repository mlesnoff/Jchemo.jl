struct Treer
    fm
end

""" 
    treer(X, y; mtry = size(X, 2),
        max_depth = -1, min_leaf = 5, min_split = 2, kwargs...)
Tree model (CART) for regression.
* `X` : X-data.
* `y` : Y-data (univariate).
* `mtry` : Nb. variables (columns) sampled in `X` at each node.
* `max_depth` : Maximum depth of the tree (default: -1, no maximum).
* `min_leaf` :  Minimum nb. observations that each leaf needs to have (default: 5).
* `min_split`: Minimum nb. observations needed for a split (default: 2).
* `kwargs` : Other named arguments to pass in function `build_tree` of `DecisionTree.jl`.

The function uses package `DecisionTree.jl' to build the tree.

## References

Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. Classication
And Regression Trees. Chapman & Hall, 1984.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245
""" 
function treer(X, y; mtry = size(X, 2),
    max_depth = -1, min_leaf = 5, min_split = 2, 
    kwargs...) 
    X = ensure_mat(X)
    y = vec(y)
    p = size(X, 2)
    mtry = min(mtry, p)
    min_purity_increase = 0
    fm = build_tree(y, X,
        mtry,
        max_depth, min_leaf, min_split,
        min_purity_increase; kwargs...)
    Treer(fm)
end

function predict(object::Treer, X)
    X = ensure_mat(X)
    m = size(X, 1)
    pred = apply_tree(object.fm, X) ;
    pred = reshape(pred, m, 1) ;
    (pred = pred,)
end

