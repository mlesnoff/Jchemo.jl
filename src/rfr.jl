""" 
    rfr(; kwargs...)
    rfr(X, y; kwargs...)
Random forest regression with DecisionTree.jl.
* `X` : X-data (n, p).
* `y` : Univariate y-data (n).
Keyword arguments:
* `n_trees` : Nb. trees built for the forest. 
* `partial_sampling` : Proportion of sampled observations for each tree.
* `n_subfeatures` : Nb. variables to select at random at each split (default: -1 ==> sqrt(#variables)).
* `max_depth` : Maximum depth of the decision trees (default: -1 ==> no maximum).
* `min_sample_leaf` : Minimum number of samples each leaf needs to have.
* `min_sample_split` : Minimum number of observations in needed for a split.
* `mth` : Boolean indicating if a multi-threading is done when new data are predicted with function `predict`.
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

The function is a wrapper of package `DecisionTree.jl' to fit a random forest regression model.

## References
Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. https://doi.org/10.1007/BF00058655

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. https://doi.org/10.1023/A:1010933404324

DecisionTree.jl https://github.com/JuliaAI/DecisionTree.jl

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, sélection de variables et applications. PhD Thesis. 
Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : trois thèmes statistiques autour de CART en 
régression (These de doctorat). Paris 11. http://www.theses.fr/2002PA112245

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)
wlst = names(X)
wl = parse.(Float64, wlst)
p = nco(X)

n_trees = 200
n_subfeatures = p / 3
max_depth = 15
model = rfr(; n_trees, n_subfeatures, max_depth) 
fit!(model, Xtrain, ytrain)
@names model
@names fitm = model.fitm

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f  
    
@names fitm.fitm
imp = fitm.fitm.featim  # variable importances
plotsp(imp', wl; xlabel = "Wavelength (nm)", ylabel = "Importance").f
```
""" 
rfr(; kwargs...) = JchemoModel(rfr, nothing, kwargs)

function rfr(X, y; kwargs...)
    par = recovkw(ParRf, kwargs).par
    X = ensure_mat(X)
    Q = eltype(X)
    y = vec(y)
    p = nco(X)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X)
        X = fscale(X, xscales)
    end
    n_subfeatures = round(Int, par.n_subfeatures)
    min_purity_increase = 0
    fitm = build_forest(y, X, 
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
    Treer(fitm, xscales, featur, par)
end

