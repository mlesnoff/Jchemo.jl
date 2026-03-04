struct Baggr
    fitm
    res_samp::NamedTuple
    q::Int
end

"""
    baggr(X, Y; fun::Function, rep = 50, rowsamp = .7, replace = false, colsamp = 1, seed = nothing, kwargs...)
    baggr(X, Y, weights::ProbabilityWeights; fun::Function, rep = 50, rowsamp = .7, replace = false, 
        colsamp = 1, seed = nothing, kwargs...)
Bagging a regression model.
* `X` : X-data (n, p).
* `Y` : Y-data (n, p).
* `colweight` : Weights (p) of the variables. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `fun` : Function defining the model.
* `rep` : Nb. of bagging replication.
* `rowsamp` : Proportion of rows sampled in `X` at each replication.
* `colsamp` : Proportion of columns sampled (without replacement) in `X` at each replication.
* `replace`: Boolean. If `false` (default), observations are sampled without replacement.
* `kwargs` : Optional named arguments to pass in 'fun`.

## References
Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. https://doi.org/10.1007/BF00058655

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. https://doi.org/10.1023/A:1010933404324

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, sélection de variables et applications. PhD Thesis. 
Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : trois thèmes statistiques autour de CART 
en régression (These de doctorat). Paris 11. http://www.theses.fr/2002PA112245

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

fitm = baggr(Xtrain, ytrain; fun = mlr, rep = 500, rowsamp = .5, colsamp = .05) ; 
#fitm = baggr(Xtrain, ytrain; fun = plskern, nlv = 15, rep = 500, rowsamp = .7, colsamp = .7) ; 
@names fitm
fitm.res_samp.srow
fitm.res_samp.srow_oob
fitm.res_samp.scol
fitm.fitm[1]
res = predict(fitm, Xtest) ; 
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", ylabel = "Observed").f 
```
""" 
function baggr(X, Y; fun::Function, rep = 50, rowsamp = .7, replace = false, colsamp = 1, kwargs...) 
    ## To do: seed
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    res_samp = sampbag(n, p; rep, rowsamp, replace, colsamp)
    srow = res_samp.srow
    scol = res_samp.scol
    fitm = list(rep)
    #@inbounds for i = 1:rep
    Threads.@threads for i = 1:rep
        fitm[i] = fun(view(X, srow[i], scol[i]), vrow(Y, srow[i]); kwargs...)
    end
    Baggr(fitm, res_samp, nco(Y))
end

function baggr(X, Y, weights::ProbabilityWeights; fun::Function, rep = 50, rowsamp = .7, replace = false, 
        colsamp = 1, kwargs...) 
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    res_samp = sampbag(n, p; rep, rowsamp, replace, colsamp)
    srow = res_samp.srow
    scol = res_samp.scol
    fitm = list(rep)
    #@inbounds for i = 1:rep
    Threads.@threads for i = 1:rep
        w = pweight(weights.values[srow[i]])
        fitm[i] = fun(view(X, srow[i], scol[i]), vrow(Y, srow[i]), w; kwargs...)
    end
    Baggr(fitm, res_samp, nco(Y))
end

"""
    predict(object::Baggr, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Baggr, X)
    X = ensure_mat(X)
    m = nro(X)
    res = similar(X, m, object.q, length(object.fitm))
    pred = similar(X, m, object.q)
    #@inbounds for k in eachindex(object.fitm)
    Threads.@threads for k in eachindex(object.fitm)
        res[:, :, k] .= predict(object.fitm[k], 
            vcol(X, object.res_samp.scol[k])).pred   # warning: @view is not accepted by XGBoost.predict
    end
    pred .= mean(res; dims = 3)
    (pred = pred,)
end

## Little slower
#function predict(object::Baggr, X)
#    rep = length(object.fitm)
#    pred = predict(object.fitm[1], X[:, object.scol[1]]).pred
#    @inbounds for i = 2:rep
#        pred .+= predict(object.fitm[i], X[:, object.scol[i]).pred
#    end
#    pred ./= rep
#    (pred = pred,)
#end

