""" 
    rfr_dt(; kwargs...)
    rfr_dt(X, y; kwargs...)
Random forest regression with DecisionTree.jl.
* `X` : X-data (n, p).
* `y` : Univariate y-data (n).
Keyword arguments:
* `n_trees` : Nb. trees built for the forest. 
* `partial_sampling` : Proportion of sampled 
    observations for each tree.
* `n_subfeatures` : Nb. variables to select at random 
    at each split (default: -1 ==> sqrt(#variables)).
* `max_depth` : Maximum depth of the decision trees 
    (default: -1 ==> no maximum).
* `min_sample_leaf` : Minimum number of samples 
    each leaf needs to have.
* `min_sample_split` : Minimum number of observations
    in needed for a split.
* `mth` : Boolean indicating if a multi-threading is 
    done when new data are predicted with function `predict`.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* Do `dump(Par(), maxdepth = 1)` to print the default 
    values of the keyword arguments. 

The function fits a random forest regression model using 
package `DecisionTree.jl'.

## References
Breiman, L., 1996. Bagging predictors. Mach Learn 24, 
123–140. https://doi.org/10.1007/BF00058655

Breiman, L., 2001. Random Forests. Machine Learning 
45, 5–32. https://doi.org/10.1023/A:1010933404324

DecisionTree.jl
https://github.com/JuliaAI/DecisionTree.jl

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
sélection de variables et applications. PhD Thesis. 
Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, 
boosting : trois thèmes statistiques autour de CART en 
régression (These de doctorat). Paris 11. 
http://www.theses.fr/2002PA112245

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
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
p = nco(X)

n_trees = 200
n_subfeatures = p / 3
max_depth = 15
mod = rfr_dt(; n_trees, 
    n_subfeatures, max_depth) ;
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)

res = predict(mod, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    
```
""" 
function rfr_dt(X, y; kwargs...)
    par = recovkwargs(Par, kwargs)
    X = ensure_mat(X)
    Q = eltype(X)
    y = vec(y)
    p = nco(X)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X)
        X = fscale(X, xscales)
    end
    n_subfeatures = Int(round(par.n_subfeatures))
    min_purity_increase = 0
    fm = build_forest(y, X, 
        n_subfeatures, 
        par.n_trees, 
        par.partial_sampling,
        par.max_depth, 
        par.min_samples_leaf,
        par.min_samples_split,
        min_purity_increase;
        #rng = Random.GLOBAL_RNG
        rng = 3
        ) 
    featur = collect(1:p)
    TreerDt(fm, xscales, featur, kwargs, par)
end

