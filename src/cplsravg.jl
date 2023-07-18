struct CplsrAvg
    fm
    fm_da::Plslda
    lev
    ni
end

"""
    cplsravg(X, Y, cla = nothing; ncla = nothing, 
        typda = "lda", nlv_da, nlv, scal::Bool = false)
Clusterwise PLSR.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `cla` : A vector (n) defining the class membership (clusters). If `cla = nothing`, 
    a random k-means clustering is done internally and returns `ncla` clusters.
* `ncla` : Only used if `cla = nothing`. 
    Number of clusters that has to be returned by the k-means clustering.
* `typda` : Type of PLSDA. Possible values are "lda" (PLS-LDA; default) or "qda" (PLS-QDA).
* `nlv_da` : Nb. latent variables (LVs) for the PLSDA.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider in the PLSR-AVG models ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 
    are averaged). Syntax such as "10" is also allowed ("10": correponds to
    the single model with 10 LVs).
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.
    The scaling is implemented for the global (distances) and local (i.e. inside
    each neighborhood) computations.

A PLSR-AVG model (see `?plsravg`) is fitted to predict Y for each of the clusters, 
and a PLS-LDA is fitted to predict, for each cluster, the probability to belong to this cluster.
The final prediction is the weighted average of the PLSR-AVG predictions, where the 
weights are the probabilities predicted by the PLS-LDA model. 

## References
Preda, C., Saporta, G., 2005. Clusterwise PLS regression on a stochastic process. 
Computational Statistics & Data Analysis 49, 99–108. https://doi.org/10.1016/j.csda.2004.05.002

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

ncla = 5 ; nlv_da = 15 ; nlv = "10:12"
fm = cplsravg(Xtrain, ytrain; 
    ncla = ncla, nlv_da = nlv_da, nlv = nlv) ;
pnames(fm)
fm.lev
fm.ni

res = Jchemo.predict(fm, Xtest) 
res.posterior
rmsep(res.pred, ytest)
plotxy(vec(res.pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f  
```
"""
function cplsravg(X, Y, cla = nothing; ncla = nothing, 
        typda = "lda", nlv_da, nlv, scal::Bool = false)
    X = ensure_mat(X) 
    Y = ensure_mat(Y)
    if isnothing(cla)
        zX = copy(X)
        if scal 
            scale!(zX, colstd(zX))
        end
        zfm = Clustering.kmeans(zX', ncla; maxiter = 1000, 
            display = :none)
        cla = zfm.assignments
    end
    ztab = tab(cla)
    lev = ztab.keys
    ni = ztab.vals
    nlev = length(lev)
    #fm_da = plsrda(X, cla; nlv = nlv_da)
    if typda == "lda"
        fm_da = plslda(X, cla; nlv = nlv_da, prior = "prop",
            scal = scal)
    elseif typda == "qda"
        fm_da = plsqda(X, cla; nlv = nlv_da, prior = "prop", 
            scal = scal)
    end
    fm = list(nlev)
    @inbounds for i = 1:nlev
        z = eval(Meta.parse(nlv))
        zmin = minimum(z)
        zmax = maximum(z)
        ni[i] <= zmin ? zmin = ni[i] - 1 : nothing
        ni[i] <= zmax ? zmax = ni[i] - 1 : nothing
        znlv = string(zmin:zmax)
        s = cla .== lev[i]
        fm[i] = plsravg(X[s, :], Y[s, :]; nlv = znlv,
            scal = scal)
    end
    CplsrAvg(fm, fm_da, lev, ni)
end

"""
    predict(object::CplrAvg, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::CplsrAvg, X)
    X = ensure_mat(X)
    m = nro(X)
    nlev = length(object.lev)
    post = predict(object.fm_da, X).posterior
    #post .= (mapreduce(i -> Float64.(post[i, :] .== maximum(post[i, :])), hcat, 1:m)')
    #post = (mapreduce(i -> mweight(exp.(post[i, :])), hcat, 1:m))'
    #post .= (mapreduce(i -> 1 ./ (1 .+ exp.(-post[i, :])), hcat, 1:m)')
    #post .= (mapreduce(i -> post[i, :] / sum(post[i, :]), hcat, 1:m))'
    acc = post[:, 1] .* predict(object.fm[1], X).pred
    @inbounds for i = 2:nlev
        if object.ni[i] >= 30
            acc .+= post[:, i] .* predict(object.fm[i], X).pred
        end
    end
    (pred = acc, posterior = post)
end


