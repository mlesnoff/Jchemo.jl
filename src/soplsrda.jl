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
    fitm_da::Union{Vector{Lda}, Vector{Qda}, Vector{Kdeda}}   
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

# Examples
```julia
```

# References
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
    transf(object::Soplsrda, Xbl)
    transf(object::Soplsrda, Xbl, nlv::Int)
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


