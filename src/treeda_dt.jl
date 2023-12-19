""" 
    treeda_dt(X, yy::Union{Array{Int}, Array{String}}; 
        n_subfeatures = 0,
        max_depth = -1, min_samples_leaf = 5, 
        min_samples_split = 2, scal::Bool = false, 
        kwargs...)
Discrimination tree (CART) with DecisionTree.jl.
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

The function fits a single discrimination tree (CART) using package 
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
using JLD2
using JchemoData
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
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
## For DA in DecisionTree.jl, 
## y must be Int or String
function treeda_dt(X, y::Union{Array{Int}, Array{String}}; 
        kwargs...) 
    par = recovkwargs(Par, kwargs)
    X = ensure_mat(X)
    Q = eltype(X)
    y = vec(y)
    p = nco(X)
    ztab = tab(y)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X)
        X = fscale(X, xscales)
    end
    n_subfeatures = Int(round(par.n_subfeatures))
    min_purity_increase = 0
    fm = build_tree(y, X,
        n_subfeatures,
        par.max_depth,
        par.min_samples_leaf,
        par.min_samples_split,
        min_purity_increase;
        #rng = Random.GLOBAL_RNG
        #rng = 3
        )
    featur = collect(1:p)
    TreedaDt(fm, xscales, featur, ztab.keys, 
        ztab.vals, kwargs, par)
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
    ## Tree
    if pnames(object.fm)[1] == :node
        pred = apply_tree(object.fm, 
            fscale(X, object.xscales))
    ## Forest 
    else
        pred = apply_forest(object.fm, 
            fscale(X, object.xscales); 
            use_multithreading = object.par.mth)
    end
    pred = reshape(pred, m, 1)
    (pred = pred,)
end

