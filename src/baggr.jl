struct Baggr3
    fm
    srow::Vector{Vector{Int64}}  # in-bag
    scol::Vector{Vector{Int64}}
    soob::Vector{Vector{Int64}}  # out-of-bag
end

""" 
    baggr(X, Y, weights = nothing, wcol = nothing; rep = 50, 
        fun, rowsamp = .7, withr = false, colsamp = 1, 
        kwargs...)
Bagging of regression models.
* `X` : X-data  (n, p).
* `Y` : Y-data  (n, q).
* `weights` : Weights (n) of the observations. Internally normalized to sum to 1.
* `wcol` : Weights (p) for the sampling of the variables.
* `rep` : Nb. of bagging repetitions.
* `fun` : Name of the function computing the model to bagg.
* `rowsamp` : Proportion of rows sampled in `X` 
    at each repetition.
* `withr`: Type of sampling of the observations
    (`true` => with replacement).
* `colsamp` : Proportion of columns sampled (without replacement) in `X` 
    at each repetition.
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
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
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

res = baggr_oob(fm, Xtrain, ytrain; score = rmsep)
res.scor

res = baggr_vi(fm, Xtrain, ytrain; score = rmsep)
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
    srow = list(rep, Vector{Int64})
    scol = list(rep, Vector{Int64})
    soob = list(rep, Vector{Int64})
    nsrow = Int64(round(rowsamp * n))
    nscol = max(1, Int64(round(colsamp * p)))
    w = similar(X, nsrow)
    zcol = collect(1:nscol) 
    zX = similar(X, nsrow, nscol)
    zY = similar(Y, nsrow, q)
    @inbounds for i = 1:rep
        # Rows
        srow[i] = sample(1:n, nsrow; replace = withr)
        soob[i] = findall(in(srow[i]).(1:n) .== 0)
        # Columns
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
        # End
        zX .= X[srow[i], scol[i]]
        zY .= Y[srow[i], :]
        if(isnothing(weights))
            fm[i] = fun(zX, zY; kwargs...)
        else
            w .= mweight(weights[srow[i]])
            fm[i] = fun(zX, zY, w; kwargs...)
        end
    end
    Baggr3(fm, srow, scol, soob)
end

function predict(object::Baggr3, X)
    rep = length(object.fm)
    # @view is not accepted by XGBoost.predict
    # @view(X[:, object.scol[i]])
    acc = predict(object.fm[1], X[:, object.scol[1]]).pred
    @inbounds for i = 2:rep
        acc .+= predict(object.fm[i], X[:, object.scol[i]]).pred
    end
    pred = acc ./ rep
    (pred = pred,)
end

