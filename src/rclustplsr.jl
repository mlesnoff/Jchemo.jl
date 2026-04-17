"""
    rclustplsr(; kwargs...)
    rclustplsr(X, Y; kwargs...)
Random clustered PLSR.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
Keyword arguments:
* `rep` : Nb. of bagging replications.
* `rowsamp` : Proportion of rows sampled in `X` at each replication.
* `replace`: Boolean. If `false` (default), observations are sampled without replacement.
* `colsamp` : Proportion of columns sampled (without replacement) in `X` at each replication.
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS used for the dimension 
    reduction before computing the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to build the clases (kmeans) and eventually to compute 
    the weights when averaging the `kavg` prototype models. Possible values are (see function `getknn`): 
    `:eucl` (Euclidean), `:mah` (Mahalanobis), `:sam` (spectral angular distance), `:cos` (cosine distance), `
    :cor` (correlation distance).
* `nproto`: Nb. prototype classes (kmeans) to build.
* `nlv` : Maximum nb. latent variables (LVs) for each prototype model.
* `K` : Nb. folds (segments) in the K-fold cross-validation. 
* `kavg` : Nb. prototype models whose predictions are averaged to compute the final prediction.
* `h` : Used when `kavg` > 1. A scalar defining the shape of the weight function computed by function `winvs`. 
    Lower is h, sharper is the function. See function `winvs` for details (keyword arguments `criw` and 
    `squared` of `winvs` can also be specified here). Used when averaging the prototype predictions.
* `scal` : Boolean. If `true`, each column of matrices X and Y of the prototype classes is 
    scaled by its uncorrected standard deviation.

Function `rclustplsr` implements a bagging of clustered PLSR (function `rclustplsr`). 
    
Basically, the pipeline mixes the principles of random forests and regression trees, but with the
following particularities:
* The 'leafs' (classes) are built by kmeans instead of trees,
* The regression model fitted on each class is a PLSR,
* Several class models can be averaged to get the final prediction.   

A number of `rep` bagging replications is run (see function `baggr`). Each replication 'b' generates a 
dataset {X(b), Y(b)} by sub-sampling rows in {`X`, `Y`} and columns in `X`. For each dataset  {X(b), Y(b)}, 
the process detailed in `protoclustplsr` is run. At the end (merge of the replications), the final prediction
is computed by the mean of the `rep` predictions.  

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

rep = 100
nlvdis = 0 ; metric = :eucl 
#nlvdis = 15 ; metric = :mah 
nproto = 3
nlv = 17
kavg = 1
fitm = rclustplsr(Xtrain, ytrain; rep, rowsamp, colsamp, nlvdis, metric, nproto, nlv, kavg, h, seed = 1234) ; 
@names fitm
fitm_bag = fitm.fitm ;
fitm_bag.res_samp.srow
fitm_bag.res_samp.srow_oob
fitm_bag.res_samp.scol
length(fitm_bag.fitm)
typeof(fitm_bag.fitm[1])
@names(fitm_bag.fitm[1])
tab(fitm_bag.fitm[1].ycla)

res = predict(fitm, Xtest)  
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", ylabel = "Observed").f
```
""" 
rclustplsr(; kwargs...) = JchemoModel(rclustplsr, nothing, kwargs)

Base.@kwdef mutable struct ParRclustplsr 
    rep::Int = 50
    rowsamp::Float64 = .7
    replace::Bool = false
    colsamp::Float64 = .7 
    nlvdis::Int = 0                          
    metric::Symbol = :eucl  
    nproto::Int = 1
    nlv::Union{Int, Vector{Int}, UnitRange} = 1
    K::Int = 5   
    kavg::Int = 1                              
    h::Float64 = Inf                        
    criw::Float64 = 4                       
    squared::Bool = false                   
    tolw::Float64 = 1e-4                    
    scal::Bool = false    
    seed::Union{Nothing, Int} = nothing
end

struct Rclustplsr
    fitm::Baggr
    par::ParRclustplsr
end

function rclustplsr(X, Y; kwargs...)
    par = recovkw(ParRclustplsr, kwargs).par 
    fitm = baggr(X, Y; fun = Jchemo.protoclustplsr, 
        rep = par.rep, 
        rowsamp = par.rowsamp,
        replace = par.replace,
        colsamp = par.colsamp,
        seed = par.seed,
        ## Start kwargs of 'fun' 
        nlvdis = par.nlvdis, 
        metric = par.metric,
        nproto = par.nproto,
        nlv = par.nlv,
        K = par.K,
        kavg = par.kavg,
        h = par.h,
        criw = par.criw,
        squared = par.squared,
        tolw = par.tolw,                    
        scal = par.scal
        )
    Rclustplsr(fitm, par)
end

function predict(object::Rclustplsr, X)
    predict(object.fitm, X)
end

