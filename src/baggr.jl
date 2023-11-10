struct Baggr
    fm
    srow::Vector{Vector{Int}}  # in-bag
    scol::Vector{Vector{Int}}
    soob::Vector{Vector{Int}}  # out-of-bag
    q
end

""" 
    baggr(X, Y, weights = nothing, wcol = nothing; rep = 50, 
        rowsamp = .7, colsamp = 1, withr = false, 
        fun, kwargs...)
Bagging of regression models.
* `X` : X-data  (n, p).
* `Y` : Y-data  (n, q).
* `weights` : Weights (n) of the observations. Internally normalized to sum to 1.
* `wcol` : Weights (p) for the sampling of the variables.
* `rep` : Nb. of bagging repetitions.
* `rowsamp` : Proportion of rows sampled in `X` 
    at each repetition.
* `colsamp` : Proportion of columns sampled (without replacement) in `X` 
    at each repetition.
* `withr`: Boolean. If `false` (default), observations are sampled without
    replacement.
* `fun` : Name of the function computing the model to bagg.
* `kwargs` : Optional named arguments to pass in 'fun`.

## References
Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. 
https://doi.org/10.1007/BF00058655

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. 
https://doi.org/10.1023/A:1010933404324

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
sélection de variables et applications. PhD Thesis. Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245

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

fm = baggr(Xtrain, ytrain; rep = 20, 
    rowsamp = .7, colsamp = .3, fun = mlr) ;
res = Jchemo.predict(fm, Xtest) ;
res.pred
rmsep(ytest, res.pred)
f, ax = scatter(vec(res.pred), ytest)
ablines!(ax, 0, 1)
f

res = oob_baggr(fm, Xtrain, ytrain; score = rmsep)
res.scor

res = vi_baggr(fm, Xtrain, ytrain; score = rmsep)
res.imp
lines(vec(res.imp), 
    axis = (xlabel = "Variable", ylabel = "Importance"))
```
""" 
function baggr(X, Y, weights = nothing, wcol = nothing; rep = 50, 
        fun, rowsamp = .7, withr = false, colsamp = 1, 
        kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    q = nco(Y)   
    fm = list(rep)
    srow = list(rep, Vector{Int})
    scol = list(rep, Vector{Int})
    soob = list(rep, Vector{Int})
    nsrow = Int(round(rowsamp * n))
    nscol = max(1, Int(round(colsamp * p)))
    w = similar(X, nsrow)
    zcol = collect(1:nscol) 
    Threads.@threads for i = 1:rep
        ## Rows
        res = samprand(n, nsrow; replace = withr)
        srow[i] = res.train
        soob[i] = res.test
        ## Columns
        if colsamp == 1
            scol[i] = zcol
        else
            if isnothing(wcol)
                scol[i] = sample(1:p, nscol; replace = false)
            else
                scol[i] = sample(1:p, StatsBase.weights(wcol), nscol; 
                    replace = false)
            end
        end
        ## End
        zsrow = srow[i]
        zscol = scol[i]
        if(isnothing(weights))
            fm[i] = fun(X[zsrow, zscol], Y[zsrow, :]; kwargs...)
        else
            w .= mweight(weights[srow[i]])
            fm[i] = fun(X[zsrow, zscol], Y[zsrow, :], w; kwargs...)
        end
    end
    Baggr(fm, srow, scol, soob, q)
end

## Little faster than the @inbounds version below
function predict(object::Baggr, X)
    X = ensure_mat(X)
    rep = length(object.fm)
    m = nro(X)
    res = similar(X, m, object.q, rep)
    pred = similar(X, m, object.q)
    Threads.@threads for i = 1:rep
        res[:, :, i] .= predict(object.fm[i], X[:, object.scol[i]]).pred
    end
    pred .= mean(res; dims = 3)
    (pred = pred,)
end

#function predict(object::Baggr, X)
#    rep = length(object.fm)
#    ## @view is not accepted by XGBoost.predict
#    ## @view(X[:, object.scol[i]])
#    pred = predict(object.fm[1], X[:, object.scol[1]]).pred
#    @inbounds for i = 2:rep
#        pred .+= predict(object.fm[i], X[:, object.scol[i]]).pred
#    end
#    pred ./= rep
#    (pred = pred,)
#end
