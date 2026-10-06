struct Baggr
    fitm
    res_samp::NamedTuple
    q::Int
end

"""
    baggr(X, Y; fun::Function, rep::Int = 50, replace::Bool = false, rowsamp::Q = .7, 
        colsamp::Q = 1., seed::Union{Nothing, Int} = nothing, kwargs...) where Q <: Float
    baggr(X, Y, weights::ProbabilityWeights; fun::Function, rep::Int = 50, replace::Bool = false, rowsamp::Q = .7, 
        colsamp::Q = 1., seed::Union{Nothing, Int} = nothing, kwargs...) where Q <: Float 
Bagging a regression model.
* `X` : X-data (n, p).
* `Y` : Y-data (n, p).
* `colweight` : Weights (p) of the variables. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `fun` : Function defining the regression model.
* `rep` : Nb. of bagging replications.
* `replace`: Boolean. If `false` (default), observations are sampled without replacement.
* `rowsamp` : Proportion of rows sampled in `X` at each replication.
* `colsamp` : Proportion of columns sampled (without replacement) in `X` at each replication.
* `seed` : Eventual seed for the `Random.MersenneTwister` generator used to select rows and columns.
* `kwargs` : Optional named arguments to pass in 'fun`.

# References
Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. https://doi.org/10.1007/BF00058655

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. https://doi.org/10.1023/A:1010933404324

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, sélection de variables et applications. PhD Thesis. 
Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : trois thèmes statistiques autour de CART 
en régression (These de doctorat). Paris 11. http://www.theses.fr/2002PA112245

# Examples
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
wlst = names(Xtrain)
wl = parse.(eltype(X[1, 1]), wlst)

rep = 200
fitm = baggr(Xtrain, ytrain; fun = mlr, rep, rowsamp = .5, colsamp = .05) ; 
#fitm = baggr(Xtrain, ytrain; fun = mlr, rep, rowsamp = .5, colsamp = .05, seed = 1234) ; 
#fitm = baggr(Xtrain, ytrain; fun = plskern, nlv = 15, rep, rowsamp = .7, colsamp = .5) ; 
#fitm = baggr(Xtrain, ytrain; fun = treer, n_subfeatures = 0, rep, rowsamp = .7, colsamp = .2) ; 
@names fitm
fitm.res_samp.srow       # indexes of the observations used as training
fitm.res_samp.srow_oob   # indexes of the oob observations
fitm.res_samp.scol       # indexes of the selected X-columns  
fitm.fitm[1]
res = predict(fitm, Xtest) ; 
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f

res = vi_baggr(fitm, Xtrain, ytrain; score = rmsep, seed = 1234) ;
@names res 
@head vi = res.vi
col = (:blue, .5)
xticks = collect(400:200:(1.1 * wl[end]))
f = Figure(size = (900, 300))
ax = Axis(f[1, 1]; xticks, xlabel = "Wavelength (nm)", ylabel = "VI")
scatter!(ax, wl, vec(vi); color = col)
lines!(ax, wl, vec(vi); color = col, linewidth = .5)
xlims!(ax, (.9 * wl[1], 1.05 * wl[end]))    
f
```
""" 
function baggr(X, Y; fun::Function, rep::Int = 50, replace::Bool = false, rowsamp::Q = .7, 
        colsamp::Q = 1., seed::Union{Nothing, Int} = nothing, kwargs...) where Q <: Float
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    res_samp = sampbag(n, p; rep, rowsamp, replace, colsamp, seed)
    srow = res_samp.srow
    scol = res_samp.scol
    fitm = list(rep)
    #@inbounds for i in eachindex(fitm)
    Threads.@threads for i in eachindex(fitm)
        fitm[i] = fun(view(X, srow[i], scol[i]), vrow(Y, srow[i]); kwargs...)
    end
    Baggr(fitm, res_samp, nco(Y))
end

function baggr(X, Y, weights::ProbabilityWeights; fun::Function, rep::Int = 50, replace::Bool = false, rowsamp::Q = .7, 
        colsamp::Q = 1., seed::Union{Nothing, Int} = nothing, kwargs...) where Q <: Float 
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    res_samp = sampbag(n, p; rep, rowsamp, replace, colsamp, seed)
    srow = res_samp.srow
    scol = res_samp.scol
    fitm = list(rep)
    #@inbounds for i = 1:rep
    Threads.@threads for i in eachindex(fitm)
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
    rep = length(object.fitm)
    res = similar(X, m, object.q, rep)
    #@inbounds for k in eachindex(object.fitm)
    Threads.@threads for k in eachindex(object.fitm)
        res[:, :, k] .= predict(
            object.fitm[k], 
            vcol(X, object.res_samp.scol[k])   # warning: @view is not accepted by XGBoost.predict
            ).pred
    end
    pred = mean(res; dims = 3)[:, :, 1]
    (pred = pred,)
end

# Little slower
#function predict(object::Baggr, X)
#    rep = length(object.fitm)
#    pred = predict(object.fitm[1], X[:, object.scol[1]]).pred
#    @inbounds for i = 2:rep
#        pred .+= predict(object.fitm[i], X[:, object.scol[i]).pred
#    end
#    pred ./= rep
#    (pred = pred,)
#end

""" 
    vi_baggr(object::Baggr, X, Y; score::Function = rmsep, seed::Union{Nothing, Int} = nothing)
Variable importance with the out-of-bag permutations method.
* `object` : Output of a bagging.
* `X` : X-data that were used in the bagging.
* `Y` : Y-data that were used in the bagging.
Keyword arguments:
* `score`: Function computing the prediction score (default is `rmsep`)
* `seed` : Eventual seed for the `Random.MersenneTwister` generator used for the random permutations.

Variable importances are computed by randomly permuting the rows of each column (successively) of the out-of-bag 
X-data (playing the role of test set), and by looking at the effect on the predictive error rate 
(e.g. RMSEP).  See function `baggr` for examples.

# References
Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. https://doi.org/10.1023/A:1010933404324

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, sélection de variables et applications. PhD Thesis. 
Université Paris Sud - Paris XI.
""" 
function vi_baggr(object::Baggr, X, Y; score::Function = rmsep, seed::Union{Nothing, Int} = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    p = nco(X)
    q = nco(Y)
    Q = eltype(X)
    res_samp = object.res_samp
    rep = length(object.fitm)        # nb rep      
    ncol = length(res_samp.scol[1])  # consistant nb. variables selected in each rep
    # Pre-alloc
    scol = similar(res_samp.scol[1], ncol)
    res = fill(Q.(NaN), p, q, rep)
    scor_ref = similar(X, 1, q)
    # End
    @inbounds for i in eachindex(object.fitm)
        srow_oob = res_samp.srow_oob[i]   # indexes of the obs being in oob 'i' (variable length)
        scol .= res_samp.scol[i]          # indexes of the variables selected for rep 'i' (consistent length) 
        m = length(srow_oob)              # nb. obs in oob 'i' (variable)
        # Prediction on X_oob 'i' (play the role of test set), and compute reference error for oob 'i'
        X_oob = view(X, srow_oob, scol)   # test set
        vpred = predict(object.fitm[i], X_oob).pred
        vY = vrow(Y, srow_oob)
        scor_ref .= score(vpred, vY)      # reference (i.e. with no permutation) for oob 'i'
        # Predictions on X_oob 'i' after permuting each column 
        # Run over all the p variables of X but compute only when variable 'j' is in an oob group
        vX = similar(X, m, p)
        @inbounds for j in axes(X, 2)  
            vseed = isnothing(seed) ? seed : seed + j - 1          
            if in(j, scol)
                # Permute rows for var 'j' (in 1:p) and compute predictions and score
                vX .= vrow(X, srow_oob)
                s = Jchemo.StatsBase.sample(MersenneTwister(vseed), 1:m, m; replace = false)      
                vX[:, j] .= vX[s, j]
                vpred .= predict(object.fitm[i], vX[:, scol]).pred
                res[j, :, i] = score(vpred, vY) - scor_ref
            end
        end
    end
    cnt = (!isnan).(res)  # nb occurences to consider for the average
    res[isnan.(res)] .= 0
    cnttot = sum(cnt, dims = 3)[:, :, 1]
    restot = sum(res, dims = 3)[:, :, 1]
    vi = restot ./ cnttot
    (vi = vi, res, cnt, restot, cnttot)
end

