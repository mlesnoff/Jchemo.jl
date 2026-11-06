Base.@kwdef mutable struct ParSoplsda{Q <: Float}  # soplsrda, soplslda
    nlv::Union{Int, Vector{Int}} = 1     
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParSoplsqda{Q <: Float}  
    nlv::Union{Int, Vector{Int}} = 1       
    prior::Union{Symbol, Vector{Q}} = :prop 
    alpha::Q = 0.   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParSoplskdeda{Q <: Float}  
    nlv::Union{Int, Vector{Int}} = 1     
    prior::Union{Symbol, Vector{Q}} = :prop 
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1.  
    scal::Symbol = :none                    
end 

struct Soplsrda{Q <: Float}
    fitm_emb::Soplsr 
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParSoplsda
end

struct Soplsprobda{Q <: Float}    # soplslda, soplsqda, soplskdeda  
    fitm_emb::Soplsr
    fitm_da::Union{Lda, Qda, Kdeda}   
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParSoplsda, ParSoplsqda, ParSoplskdeda}
end

"""
    soplsrda(; kwargs...)
    soplsrda(Xbl, y; kwargs...)
    soplsrda(Xbl::Vector{Matrix{Q}}, y::Vector{String}, weights::ProbabilityWeights{Q}; 
        kwargs...) where Q <: Float
Discrimination by multiblock sequentially orthorgonalized Plsr (SOPLSR-DA).
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` from data (n, p).  
* `y` : Univariate class membership (n). Must be a `Vector{String}`.
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs; = scores) to compute. Can be a single integer (same nb. LVs for each block) 
    or a vector of integers (nb. LVs for each block).
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Symbol defining the column scaling of blocks in `Xbl` and Ydummy (before the block scaling). 
    Possible values are: `:none` (default), `:std` (uncorrected STD), `:prt` (pareto) and `:mad` (MAD).

Same as PLSR-DA but the Ydummy (class dummy table Ydummy) predictions are computed from SOPLSR.

# Examples
```julia
```

# References
Biancolillo, A., 2016. Method development in the area of multi-block analysis focused on food analysis. 
PhD. University of Copenhagen.

Biancolillo, A., Måge, I., Næs, T., 2015. Combining SO-PLS and linear discriminant analysis 
for multi-block classification. Chemometrics and Intelligent Laboratory Systems 141, 58–67. 
https://doi.org/10.1016/j.chemolab.2014.12.001

Menichelli et al., 2014. SO-PLS as an exploratory tool for path modelling. 
Food Quality and Preference, 36, 122-134.
"""
soplsrda(; kwargs...) = JchemoModel(soplsrda, nothing, kwargs)

function soplsrda(Xbl, y; kwargs...)
    Xbl = ensure_mat_mb(Xbl)
    y = vec(y)
    Q = eltype(Xbl[1])
    prior = recovkw(ParSoplsda{Q}, kwargs).par.prior
    weights = pweightcla(Q, y; prior)
    Soplsrda(Xbl, y, weights; kwargs...)
end

function soplsrda(Xbl::Vector{Matrix{Q}}, y::Vector{String}, weights::ProbabilityWeights{Q}; 
        kwargs...) where Q <: Float
    par = recovkw(ParSoplsda{Q}, kwargs).par
    res = dummy(Q, y)
    ni = tab(y).n
    priors = aggsumv(weights.values, y).val  # output not used, only for information
    fitm_emb = soplsr(Xbl, res.Y, weights; kwargs...)
    par.nlv = fitm_emb.par.nlv
    Soplsrda(fitm_emb, ni, priors, res.lev, par)
end

""" 
    transf(object::Union{Soplsrda, Soplsprobda}, Xbl)
Compute latent variables (LVs; = scores) from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) of X-data for which LVs are computed.
* `nlv` : Nb. LVs to compute.
""" 
transf(object::Union{Soplsrda, Soplsprobda}, Xbl) = transf(object.fitm_emb, Xbl)

"""
    predict(object::Soplsrda, Xbl)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) of X-data for which predictions are computed.
""" 
function predict(object::Soplsrda, Xbl)
    m = nro(Xbl[1])
    res = predict(object.fitm_emb, Xbl)
    v =  mapslices(argmax, res.pred; dims = 2)  # if equal, argmax takes the first
    pred = reshape(recod_indbylev(vec(v), object.lev), m, 1)
    (pred = pred, posterior = res.pred)
end


