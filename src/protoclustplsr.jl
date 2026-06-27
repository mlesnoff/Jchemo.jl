"""
    protoclustplsr(; kwargs...)
    protoclustplsr(X, Y; kwargs...)
Clustered PLSR.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
Keyword arguments:
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS used for the dimension 
    reduction before computing the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to build the classes (kmeans) and to compute the weights when averaging 
    the `kavg` prototype models. Possible values are (see function `getknn`): `:eucl` (Euclidean), `:mah` (Mahalanobis), 
    `:sam` (spectral angular distance), `:cos` (cosine distance), `:cor` (correlation distance).
* `nproto`: Nb. prototypes to define.
* `nlv` : Maximum nb. latent variables (LVs) for each PLSR prototype model.
* `K` : Nb. folds (segments) in the K-fold cross-validation. 
* `kavg` : Nb. prototype models whose predictions are averaged to compute the final prediction.
* `h` : A scalar defining the shape of the weight function used to average the prototype predictions. The weights are 
    computed by function `winvs`)Lower is h, sharper is the function. See function `winvs` for details (keyword arguments
    `criw` and `squared` of `winvs` can also be specified here).
* `scal` : Boolean. If `true`, each column of matrices X and Y of the prototype neighborhood is scaled by its 
    uncorrected standard deviation before implementing the PLSR.
* `seed` : Eventual seed for the `Random.MersenneTwister` generator, used to initialize the kmeans algorithm. 

Function `protoclustplsr` implements an averaging of prototype PLSR models.

*Distance computations*
* In the actual version of the function, the dissimilarities between observations are computed on the original X-data or 
or on global PLS scores computed from (`X`, `Y`). This is managed by argument `nlvdis`. 
* Argument `metric` defines the type of dissimilarity used.

*Model fitting*
* A number of `nproto` classes (mutually exclusive) are built by kmeans. Each class centroid defines a 'prototype'. 
    The set of `nproto` classes is assumed to represent the data heterogeneity (diversity of application domains 
    present in the data).
* On each class, a PLSR is optimized using a K-fold cross-validation and stored. This defines the prototype model.

*Prediction*
* Each new observation to predict is assigned to its `kavg` nearest prototypes.
* The final prediction is computed by a weighted average of the `kavg` predictions of the corresponding 
    prototype models. The weighting is computed from the distances between the new observation and the `kavg` prototype centers 
    (function `winvs`). TThe weights decrease with the distance (decreasing weighting function). If `kavg = 1`, only one PLSR model 
    is used (the closest prototype) and there is no averaging.

The kmeans step is done with package `Clustering.jl` (https://github.com/JuliaStats/Clustering.jl).

Notes: 
* This pipeline is still under construction, some details could change in the future.
* The actual version of the function works for multivariate `Y` but the PLSR optimizations are done only based on the
    first Y column (this will be fixed later). 

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

nlvdis = 0 ; metric = :eucl 
#nlvdis = 15 ; metric = :mah 
nproto = 3
nlv = 17
kavg = 1
model = protoclustplsr(; nlvdis, metric, nproto, nlv, kavg, seed = 1234) ; 
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm 
@names fitm = model.fitm
tab(fitm.ycla)
@names fitm.fitm
fitm.fitm.coefs

res = predict(model, Xtest) ; 
@names res 
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",  
    ylabel = "Observed").f  
```
""" 
Base.@kwdef mutable struct ParProtoclustplsr 
    nlvdis::Signed = 0    # To do                         
    metric::Symbol = :eucl  
    nproto::Signed = 1
    nlv::Signed = 1
    K::Signed = 5   
    kavg::Signed = 1                              
    h::Float64 = Inf                        
    criw::Float64 = 4.                       
    squared::Bool = false                   
    tolw::Float64 = 1e-4                    
    scal::Symbol = :none    
    seed::Union{Nothing, Int} = nothing
end

struct Protoclustplsr
    fitm::Protoyclaplsr
    fitm_emb::Union{Nothing, Plsr}
    fitm_clust::Clustering.KmeansResult
    ycla::AbstractVector
    par::ParProtoclustplsr
end

protoclustplsr(; kwargs...) = JchemoModel(protoclustplsr, nothing, kwargs)

function protoclustplsr(X, Y; kwargs...)
    par = recovkw(ParProtoclustplsr{Q}, kwargs).par 
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    if par.nlvdis == 0
        fitm_emb = nothing
        zX = copy(X)
    else
        fitm_emb = plskern(X, Y; nlv = par.nlvdis)
        zX = fitm_emb.T
    end
    if par.metric == :eucl
        distance = Distances.Euclidean()
    elseif par.metric == :mah
        distance = Distances.Mahalanobis(covm(zX))
    elseif par.metric == :cos
        distance = Jchemo.CosDist()
    elseif par.metric == :sam
        distance = Jchemo.SamDist()
    elseif par.metric == :cor 
        distance = Jchemo.CorDist()
    #elseif par.metric == :was
    #    distance =  Jchemo.CorDist_b() # Jchemo.WasDist()
    end
    fitm_clust = kmeans(zX', par.nproto; 
        init = :kmpp,    # default
        maxiter = 5000, 
        display = :none,
        distance,
        rng = Random.MersenneTwister(par.seed)
        ) 
    ycla = fitm_clust.assignments
    fitm = Jchemo.protoyclaplsr(X, Y, ycla; metric = par.metric, nlv = par.nlv, K = par.K, kavg = par.kavg, 
        h = par.h, criw = par.criw, squared = par.squared, tolw = par.tolw, scal = par.scal) 
    Protoclustplsr(fitm, fitm_emb, fitm_clust, ycla, par)   
end

function predict(object::Protoclustplsr, X)
    Jchemo.predict(object.fitm, X) 
end



