""" 
    treeda_dt(X, yy::Union{Array{Int64}, Array{String}}; 
        n_subfeatures = 0,
        max_depth = -1, min_samples_leaf = 5, 
        min_samples_split = 2, scal = false, 
        kwargs...)
Discrimination tree (CART) with DecisionTree.jl.
* `X` : X-data (n obs., p variables).
* `y` : Univariate y-data (n obs.).
* `n_subfeatures` : Nb. variables to select at random (default: 0 ==> keep all).
* `max_depth` : Maximum depth of the decision tree (default: -1 ==> no maximum).
* `min_sample_leaf` : Minimum number of samples each leaf needs to have.
* `min_sample_split` : Minimum number of observations in needed for a split.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* `kwargs` : Optional named arguments to pass in function `build_tree` 
    of `DecisionTree.jl`.

The function fits a single discrimination tree (CART) using package 
`DecisionTree.jl'.

## References
Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. Classification
And Regression Trees. Chapman & Hall, 1984.

DecisionTree.jl
https://github.com/dmlc/XGBoost.jl
https://zenodo.org/record/7359268#.ZBswGHbMK3A

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245

## Examples
```julia
using JchemoData, JLD2

mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "iris.jld2") 
@load db dat
pnames(dat)
summ(dat.X)
  
X = dat.X[:, 1:4] 
y = dat.X[:, 5]
ntot, p = size(X)
  
ntrain = 120
s = sample(1:n, ntrain; replace = false) 
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)
ntest = ntot - ntrain
(ntot = ntot, ntrain, ntest)

tab(ytrain)
tab(ytest)

n_subfeatures = 2 
max_depth = 6
fm = treeda_dt(Xtrain, ytrain;
    n_subfeatures = n_subfeatures, max_depth = max_depth) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest)
res.pred
err(res.pred, ytest) 
```
""" 
function treeda_dt(X, y::Union{Array{Int64}, Array{String}}; 
        n_subfeatures = 0,
        max_depth = -1, min_samples_leaf = 5, 
        min_samples_split = 2, scal = false, 
        kwargs...) 
    X = ensure_mat(X)
    y = vec(y)
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
    ztab = tab(y)
    lev = ztab.keys 
    ni = ztab.vals 
    n_subfeatures = Int64(round(n_subfeatures))
    min_purity_increase = 0
    fm = build_tree(y, X,
        n_subfeatures,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase;
        kwargs...
        #rng = Random.GLOBAL_RNG
        #rng = 3
        )
    featur = collect(1:p)
    mth = true
    TreedaDt(fm, xscales, featur, lev, ni, mth)
end

""" 
    rfda_dt(X, yy::Union{Array{Int64}, Array{String}}; 
        n_trees = 10,
        partial_sampling = .7,  
        n_subfeatures = -1,
        max_depth = -1, min_samples_leaf = 5, 
        min_samples_split = 2, scal = false, 
        mth = true, kwargs...)
Random forest discrimination with DecisionTree.jl.
* `X` : X-data (n obs., p variables).
* `y` : Univariate Y-data (n obs.).
* `n_trees` : Nb. trees built for the forest. 
* `partial_sampling` : Proportion of sampled observations for each tree.
* `n_subfeatures` : Nb. variables to select at random at each split (default: 1 ==> sqrt(#variables)).
* `max_depth` : Maximum depth of the decision trees (default: -1 ==> no maximum).
* `min_sample_leaf` : Minimum number of samples each leaf needs to have.
* `min_sample_split` : Minimum number of observations in needed for a split.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* `mth` : Boolean indicating if a multi-threading is done when new data are 
    predicted with function `predict`.
* `kwargs` : Optional named arguments to pass in function `build_forest` 
    of `DecisionTree.jl`.

The function fits a random forest discrimination model using package 
`DecisionTree.jl'.

## References
Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. 
https://doi.org/10.1007/BF00058655

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. 
https://doi.org/10.1023/A:1010933404324

DecisionTree.jl
https://github.com/dmlc/XGBoost.jl
https://zenodo.org/record/7359268#.ZBswGHbMK3A

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
sélection de variables et applications. PhD Thesis. Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245

## Examples
```julia
using JchemoData, JLD2

mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "iris.jld2") 
@load db dat
pnames(dat)
summ(dat.X)
  
X = dat.X[:, 1:4] 
y = dat.X[:, 5]
ntot, p = size(X)
  
ntrain = 120
s = sample(1:n, ntrain; replace = false) 
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)
ntest = ntot - ntrain
(ntot = ntot, ntrain, ntest)

tab(ytrain)
tab(ytest)

n_subfeatures = 2 
fm = rfda_dt(Xtrain, ytrain; n_trees = 100,
    n_subfeatures = n_subfeatures) ;
pnames(fm)

res = Jchemo.predict(fm, Xtest)
res.pred
err(res.pred, ytest) 
```
""" 
function rfda_dt(X, y::Union{Array{Int64}, Array{String}}; 
        n_trees = 10,
        partial_sampling = .7,  
        n_subfeatures = -1,
        max_depth = -1, min_samples_leaf = 5, 
        min_samples_split = 2, scal = false, 
        mth = true, kwargs...)
    X = ensure_mat(X)
    y = vec(y)
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
    ztab = tab(y)
    lev = ztab.keys 
    ni = ztab.vals 
    n_subfeatures = Int64(round(n_subfeatures))
    min_purity_increase = 0
    fm = build_forest(y, X, 
        n_subfeatures, 
        n_trees, 
        partial_sampling,
        max_depth, 
        min_samples_leaf,
        min_samples_split,
        min_purity_increase;
        kwargs...
        #rng = Random.GLOBAL_RNG
        #rng = 3
        ) 
    featur = collect(1:p)
    TreedaDt(fm, xscales, featur, lev, ni, mth)
end


"""
    predict(object::TreedaDt, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::TreedaDt, X)
    X = ensure_mat(X)
    m = nro(X)
    if pnames(object.fm)[1] == :node
        pred = apply_tree(object.fm, scale(X, object.xscales))
    else
        pred = apply_forest(object.fm, scale(X, object.xscales); 
            use_multithreading = object.mth)
    end
    pred = reshape(pred, m, 1)
    (pred = pred,)
end





