"""
    rclustplsr(; kwargs...)
    rclustplsr(X, Y; kwargs...)
Random clutered PLSR.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
Keyword arguments:
* `rep` : Nb. of bagging replication.
* `rowsamp` : Proportion of rows sampled in `X` at each replication.
* `colsamp` : Proportion of columns sampled (without replacement) in `X` at each replication.
* `replace`: Boolean. If `false` (default), observations are sampled without replacement.
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS used for the dimension 
    reduction before computing the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors and to compute the weights 
    (see function `getknn`). Possible values are: `:eucl` (Euclidean), `:mah` (Mahalanobis), 
    `:sam` (spectral angular distance), `:cos` (cosine distance), `:cor` (correlation distance).
* `nproto`: Nb. prototypes selected.
* `nlv` : Maximum nb. latent variables (LVs) for each prototype model.
* `kavg` : Nb. prototype models whose predictions are averaged to compute the final prediction.
* `h` : A scalar defining the shape of the weight function computed by function `winvs`. Lower is h, 
    sharper is the function. See function `winvs` for details (keyword arguments `criw` and `squared` of 
    `winvs` can also be specified here). Used when averaging the prototype predictions.
* `scal` : Boolean. If `true`, ecah column of matrices X and Y of the prototype neighborhood is 
    scaled by its uncorrected standard deviation.

Function `rclustplsr` implements a kNN-averaging of prototype-PLSR models.

*Model fitting*
* A number of `nproto` observations (x, y), referred to as 'prototypes', are sampled in the training data. 
    In the actual version of the function, the sampling is done on `X` or on global PLS scores computed 
    from (`X`, `Y`). 
* A neighborhood is selected around each prototype. The prototype neighborhoods are assumed to represent 
    the data variability, in particular, a representative diversity of application domains.
    (Note: A same observation can eventually belong to several neighborhoods).
* On each prototype neighborhood, a PLSR (= prototype model) is optimized using a K-fold cross-validation
   (K = 3), and stored.

*Prediction*
* Each new observation to predict is assigned to its `kavg` nearest prototypes, based on its distances 
    to the prototype centers.
* The final prediction is computed by a weighted average of the `kavg` predictions of the corresponding 
  prototype models. The weighting is computed from the relative distances between the new observation and 
  the `kavg` prototype centers (function `winvs`). 

Note: 
* This pipeline is still under construction, some details could change in the future.

## Examples
```julia
```
""" 
rclustplsr(; kwargs...) = JchemoModel(rclustplsr, nothing, kwargs)

Base.@kwdef mutable struct ParRclustPlsr 
    rep::Int = 50
    rowsamp::Float64 = .7
    replace::Bool = false
    colsamp::Float64 = .7
    #nlvdis::Int = 0    # To do                         
    metric::Symbol = :eucl  
    nproto::Int = 1
    nlv::Union{Int, Vector{Int}, UnitRange} = 1   
    kavg::Int = 1                              
    h::Float64 = Inf                        
    criw::Float64 = 4                       
    squared::Bool = false                   
    tolw::Float64 = 1e-4                    
    scal::Bool = false    
end

struct RclustPlsr
    fitm::Baggr
    par::ParRclustPlsr
end

function rclustplsr(X, Y; kwargs...)
    par = recovkw(ParRclustPlsr, kwargs).par 
    fitm = baggr(X, Y; fun = Jchemo.protoclustplsr, 
        rep = par.rep, 
        rowsamp = par.rowsamp,
        replace = par.replace,
        colsamp = par.colsamp,
        ## 'fun' kwargs 
        #nlvdis = par.colsamp,  # To do
        metric = par.metric,
        nproto = par.nproto,
        nlv = par.nlv,
        kavg = par.kavg,
        h = par.h,
        criw = par.criw,
        squared = par.squared,
        tolw = par.tolw,                    
        scal = par.scal
        )
    RclustPlsr(fitm, par)
end

function predict(object::RclustPlsr, X)
    predict(object.fitm, X)
end



