#""" 
#    baggr(X, Y, weights = nothing, colweight = nothing; rep = 50, rowsamp = .7, colsamp = 1, replace = false, 
#        fun, kwargs...)
#Bagging of regression models.
#* `X` : X-data  (n, p).
#* `Y` : Y-data  (n, q).
#* `weights` : Weights (n) of the observations. Internally normalized to sum to 1.
#* `colweight` : Weights (p) for the sampling of the variables.
#* `rep` : Nb. of bagging repetitions.
#* `rowsamp` : Proportion of rows sampled in `X` at each repetition.
#* `colsamp` : Proportion of columns sampled (without replacement) in `X` at each repetition.
#* `replace`: Boolean. If `false` (default), observations are sampled without replacement.
#* `fun` : Name of the function computing the model to bagg.
#* `kwargs` : Optional named arguments to pass in 'fun`.

## References
#Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. 
#https://doi.org/10.1007/BF00058655

#Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. 
#https://doi.org/10.1023/A:1010933404324

#Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
#sélection de variables et applications. PhD Thesis. Université Paris Sud - Paris XI.

#Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
#trois thèmes statistiques autour de CART en régression (These de doctorat). 
#Paris 11. http://www.theses.fr/2002PA112245

### Examples
#```julia
#using JchemoData, JLD2, CairoMakie
#path_jdat = dirname(dirname(pathof(JchemoData)))
#db = joinpath(path_jdat, "data/cassav.jld2") 
#@load db dat
#pnames(dat)

#X = dat.X 
#y = dat.Y.tbc
#year = dat.Y.year
#tab(year)
#s = year .<= 2012
#Xtrain = X[s, :]
#ytrain = y[s]
#Xtest = rmrow(X, s)
#ytest = rmrow(y, s)

#fitm= baggr(Xtrain, ytrain; rep = 20, 
#    rowsamp = .7, colsamp = .3, fun = mlr) ;
#res = Jchemo.predict(fitm, Xtest) ;
#res.pred
#rmsep(ytest, res.pred)
#f, ax = scatter(vec(res.pred), ytest)
#ablines!(ax, 0, 1)
#f

#res = oob_baggr(fitm, Xtrain, ytrain; score = rmsep)
#res.scor

#res = vi_baggr(fitm, Xtrain, ytrain; score = rmsep)
#res.imp
#lines(vec(res.imp), 
#    axis = (xlabel = "Variable", ylabel = "Importance"))
#```
#""" 
struct Baggr
    fitm
    res_samp::NamedTuple
    q::Int
end

function baggr(X, Y; fun::Function, rep = 50, rowsamp = .7, replace = false, colsamp = 1, seed = nothing, kwargs...) 
    res_samp = Jchemo.sampbag(X; rep, rowsamp, replace, colsamp)
    srow = res_samp.srow
    scol = res_samp.scol
    fitm = list(rep)
    @inbounds for i = 1:rep
    #Threads.@threads for i = 1:rep
        fitm[i] = fun(X[srow[i], scol[i]], Y[srow[i], :]; kwargs...)
    end
    Baggr(fitm, res_samp, nco(Y))
end

function baggr(X, Y, weights::Jchemo.ProbabilityWeights; fun::Function, rep = 50, rowsamp = .7, replace = false, 
        colsamp = 1, seed = nothing, kwargs...) 
    res_samp = Jchemo.sampbag(X; rep, rowsamp, replace, colsamp)
    srow = res_samp.srow
    scol = res_samp.scol
    fitm = list(rep)
    @inbounds for i = 1:rep
    #Threads.@threads for i = 1:rep
        w = pweight(weights.values[srow[i]])
        fitm[i] = fun(X[srow[i], scol[i]], Y[srow[i], :], w)
    end
    Baggr(fitm, res_samp, nco(Y))
end

## Little faster than the @inbounds version below
function predict(object::Baggr, X)
    X = ensure_mat(X)
    m = nro(X)
    res = similar(X, m, object.q, length(object.fitm))
    pred = similar(X, m, object.q)
    @inbounds for k in eachindex(object.fitm)
    #Threads.@threads for k in eachindex(object.fitm)
        res[:, :, k] .= predict(object.fitm[k], X[:, object.res_samp.scol[k]]).pred
    end
    pred .= mean(res; dims = 3)
    (pred = pred,)
end

#function predict(object::Baggr, X)
#    rep = length(object.fitm)
#    ## @view is not accepted by XGBoost.predict
#    ## @view(X[:, object.scol[i])
#    pred = predict(object.fitm[1], X[:, object.scol[1]]).pred
#    @inbounds for i = 2:rep
#        pred .+= predict(object.fitm[i], X[:, object.scol[i]).pred
#    end
#    pred ./= rep
#    (pred = pred,)
#end


