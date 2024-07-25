""" 
    rfda(X, y; kwargs...)
Random forest discrimination with DecisionTree.jl.
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
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

The function fits a random forest discrimination² model using 
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
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
pnames(dat)
X = dat.X
Y = dat.Y
n, p = size(X) 
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = n, ntrain, ntest)
tab(ytrain)
tab(ytest)

n_trees = 200
n_subfeatures = p / 3 
max_depth = 10
mod = model(rfda; n_trees, n_subfeatures, max_depth) 
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
fm = mod.fm ;
fm.lev
fm.ni

res = predict(mod, Xtest) ; 
pnames(res) 
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt
```
""" 
function rfda(X, y::Union{Array{Int}, Array{String}}; kwargs...)
    ## For DA in DecisionTree.jl, 
    ## y must be Int or String
    par = recovkw(ParRf, kwargs).par
    X = ensure_mat(X)
    Q = eltype(X)
    y = vec(y)
    p = nco(X)
    taby = tab(y)
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
        #rng = 3
        ) 
    featur = collect(1:p)
    Treeda(fm, xscales, featur, taby.keys, taby.vals, par)
end
