struct TreerDt
    fm
    xscales::Vector{Float64}
    featur::Vector{Int64}
    mth::Bool 
end

struct TreedaDt 
    fm
    xscales::Vector{Float64}
    featur::Vector{Int64}
    lev::AbstractVector
    ni::AbstractVector
    mth::Bool 
end

""" 
    treer_dt(X, y; n_subfeatures = 0,
        max_depth = -1, min_samples_leaf = 5, 
        min_samples_split = 2, scal::Bool = false, 
        kwargs...)
Regression tree (CART) with DecisionTree.jl.
* `X` : X-data (n obs., p variables).
* `y` : Univariate y-data (n obs.).
* `n_subfeatures` : Nb. variables to select at random at each split (default: 0 ==> keep all).
* `max_depth` : Maximum depth of the decision tree (default: -1 ==> no maximum).
* `min_sample_leaf` : Minimum number of samples each leaf needs to have.
* `min_sample_split` : Minimum number of observations in needed for a split.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* `kwargs` : Optional named arguments to pass in function `build_tree` 
    of `DecisionTree.jl`.

The function fits a single regression tree (CART) using package 
`DecisionTree.jl'.

## References
Breiman, L., Friedman, J. H., Olshen, R. A., and Stone, C. J. Classification
And Regression Trees. Chapman & Hall, 1984.

DecisionTree.jl
https://github.com/JuliaAI/DecisionTree.jl

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245

## Examples
```julia
using JchemoData, JLD2, CairoMakie

path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/challenge2021.jld2")
@load db dat
pnames(dat)

Xtrain = dat.Xtrain
Ytrain = dat.Ytrain
ytrain = Ytrain.y
s = dat.Ytest.inst .== 1 
Xtest = dat.Xtest[s, :]
Ytest = dat.Ytest[s, :]
ytest = Ytest.y
wl = names(Xtrain) 
wl_num = parse.(Float64, wl) 
ntrain, p = size(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

f = 21 ; pol = 3 ; d = 2 
Xptrain = savgol(snv(Xtrain); f, pol, d) 
Xptest = savgol(snv(Xtest); f, pol, d) 

n_subfeatures = p / 3 
max_depth = 20
fm = treer_dt(Xptrain, ytrain; 
    n_subfeatures = n_subfeatures, 
    max_depth = max_depth) ;
pnames(fm)

res = Jchemo.predict(fm, Xptest)
res.pred
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f  
```
""" 
function treer_dt(X, y; n_subfeatures = 0,
        max_depth = -1, min_samples_leaf = 5, 
        min_samples_split = 2, scal::Bool = false, 
        kwargs...) 
    X = ensure_mat(X)
    y = Float64.(vec(y))
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
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
    TreerDt(fm, xscales, featur, mth)
end

""" 
    rfr_dt(X, y; n_trees = 10,
        partial_sampling = .7,  
        n_subfeatures = -1,
        max_depth = -1, min_samples_leaf = 5, 
        min_samples_split = 2, scal::Bool = false, 
        mth = true, kwargs...)
Random forest regression with DecisionTree.jl.
* `X` : X-data (n obs., p variables).
* `y` : Univariate y-data (n obs.).
* `n_trees` : Nb. trees built for the forest. 
* `partial_sampling` : Proportion of sampled observations for each tree.
* `n_subfeatures` : Nb. variables to select at random at each split (default: -1 ==> sqrt(#variables)).
* `max_depth` : Maximum depth of the decision trees (default: -1 ==> no maximum).
* `min_sample_leaf` : Minimum number of samples each leaf needs to have.
* `min_sample_split` : Minimum number of observations in needed for a split.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* `mth` : Boolean indicating if a multi-threading is done when new data are 
    predicted with function `predict`.
* `kwargs` : Optional named arguments to pass in function `build_forest` 
    of `DecisionTree.jl`.

The function fits a random forest regression model using package 
`DecisionTree.jl'.

## References
Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. 
https://doi.org/10.1007/BF00058655

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. 
https://doi.org/10.1023/A:1010933404324

DecisionTree.jl
https://github.com/JuliaAI/DecisionTree.jl

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
sélection de variables et applications. PhD Thesis. Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245

## Examples
```julia
using JchemoData, JLD2, CairoMakie

path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/challenge2021.jld2")
@load db dat
pnames(dat)

Xtrain = dat.Xtrain
Ytrain = dat.Ytrain
ytrain = Ytrain.y
s = dat.Ytest.inst .== 1 
Xtest = dat.Xtest[s, :]
Ytest = dat.Ytest[s, :]
ytest = Ytest.y
wl = names(Xtrain) 
wl_num = parse.(Float64, wl) 
ntrain, p = size(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

f = 21 ; pol = 3 ; d = 2 
Xptrain = savgol(snv(Xtrain); f, pol, d) 
Xptest = savgol(snv(Xtest); f, pol, d) 

n_subfeatures = p / 3 
fm = rfr_dt(Xptrain, ytrain; n_trees = 100,
    n_subfeatures = n_subfeatures) ;
pnames(fm)

res = Jchemo.predict(fm, Xptest)
res.pred
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f  
```
""" 
function rfr_dt(X, y; n_trees = 10,
        partial_sampling = .7,  
        n_subfeatures = -1,
        max_depth = -1, min_samples_leaf = 5, 
        min_samples_split = 2, scal::Bool = false, 
        mth = true, kwargs...)
    X = ensure_mat(X)
    y = Float64.(vec(y))
    p = nco(X)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        X = scale(X, xscales)
    end
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
    TreerDt(fm, xscales, featur, mth)
end


"""
    predict(object::TreerDt, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::TreerDt, X)
    X = ensure_mat(X)
    m = nro(X)
    if pnames(object.fm)[1] == :node
        pred = apply_tree(object.fm, scale(X, object.xscales))
    else
        pred = apply_forest(object.fm, scale(X, object.xscales); 
            use_multithreading = object.mth)
    end
    pred = reshape(pred, m, 1)
    (pred = Float64.(pred),)
end





