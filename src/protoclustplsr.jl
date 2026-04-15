"""
    protoclustplsr(; kwargs...)
    protoclustplsr(X, Y; kwargs...)
Random clustered PLSR.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
Keyword arguments:
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

Function `protoclustplsr` implements a 'random clustered PLSR'. Basically, the pipeline mixes the principles of
random forests and regression trees, but with the following particularities:
* The 'leafs' (classes) are built by kmeans instead of trees,
* The regression model fitted on each class is a PLSR,
* Several class models can be averaged to get the final prediction.   

A number of `rep` bagging replications is run in the same way as in random forests. Each replication 'b' 
generates a dataset {X(b), Y(b)} by sub-sampling rows in {`X`, `X`} and columns in `X`. For each dataset 
{X(b), Y(b)}, the process detailed below is run.

*Model fitting for replication 'b'*
* A number of `nproto` classes ( mutually exclusive) are built by kmeans on X(b). Each class centroid 
    defines a 'prototype'. The `nproto` classes are assumed to represent the data heterogeneity (diversity 
    of application domains).
* On each class, a PLSR is optimized using a K-fold cross-validation on {X(b), Y(b)} and stored. This defines 
    the prototype model.

*Prediction for replication 'b'*
* Each new observation to predict is assigned to its `kavg` nearest prototypes.
* The final prediction is computed by a weighted average of the `kavg` predictions of the corresponding 
    prototype models. The weighting is computed from the relative distances between the new observation and 
    the `kavg` prototype centers (function `winvs`). If `kavg = 1`, only one PLSR model is used (the closest
    prototype) and there is no averaging.

At the end (merge of the replications), the final prediction is computed by the mean of the `rep` 
predictions.  

The kmeans step is done with package `Clustering.jl` (https://github.com/JuliaStats/Clustering.jl).

Notes: 
* This pipeline is still under construction, some details could change in the future.
* The actual version of the function works for multivariate `Y` but the PLSR optimizations
    are done only based on the first Y column (this will be fixed later). 

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

nproto = 3
metric = :cos
nlv = 15
kavg = 1
fitm = protoclustplsr(Xtrain, ytrain; rep = 100, metric, nproto, nlv, kavg) ; 
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
Base.@kwdef mutable struct Parprotoclustplsr 
    nlvdis::Int = 0    # To do                         
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

struct Protoclustplsr
    fitm::ProtoYclaPlsr
    fitm_clust::Clustering.KmeansResult
    ycla::AbstractVector
end

protoclustplsr(; kwargs...) = JchemoModel(protoclustplsr, nothing, kwargs)

function protoclustplsr(X, y; metric = :eucl, nproto, nlv, K = 5, kavg = 1, h = 1, criw = 4, squared = false, 
        tolw = 1e-4, scal = false)
    par = recovkw(Parprotoclustplsr, kwargs).par 
    if par.metric == :eucl
        distance = Distances.Euclidean()
    elseif par.metric == :cos
        distance = Jchemo.CosDist()
    elseif par.metric == :sam
        distance = Jchemo.SamDist()
    elseif par.metric == :cor 
        distance = Jchemo.CorDist()
    #elseif par.metric == :was
    #    distance =  Jchemo.CorDist_b() # Jchemo.WasDist()
    end
    fitm_clust = kmeans(X', nproto; 
        init = :kmpp,    # default
        maxiter = 5000, 
        display = :none,
        distance = distance,
        rng = Random.MersenneTwister(seed)
        ) 
    ycla = fitm_clust.assignments
    fitm = Jchemo.protoyclaplsr(X, y, ycla; metric = par.metric, nlv = par.nlv, K = par.K, kavg = par.kavg, h = par.h,
        criw = par.criw, squared = par.squared, tolw = par.tolw, scal = par.scal) 
    Protoclustplsr(fitm, fitm_clust, ycla) 
end

function predict(object::Protoclustplsr, X)
    predict(object.fitm, X) 
end



