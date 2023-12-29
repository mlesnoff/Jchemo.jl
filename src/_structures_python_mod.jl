Base.@kwdef mutable struct Transformer{
        FUN <: Function, 
        FIT <: Function, 
        TRANSF <: Function,
        FM, KWARGS <: Base.Pairs}
    fun::FUN
    fit::FUN
    transf::FUN
    fm::Union{Nothing, FM}
    kwargs::KWARGS    
end
Base.@kwdef mutable struct Predictor{
        FUN <: Function, 
        FIT <: Function, 
        TRANSF <: Function,
        PREDICT <: Function, 
        FM, KWARGS <: Base.Pairs}
    fun::FUN
    fit::FUN
    transf::Union{Nothing, FUN}
    predict::FUN
    fm::Union{Nothing, FM}
    kwargs::KWARGS    
end
Base.@kwdef mutable struct PredictorNoY{
        FUN <: Function, 
        FIT <: Function, 
        PREDICT <: Function, 
        FM, KWARGS <: Base.Pairs}
    fun::FUN
    fit::FUN
    predict::FUN
    fm::Union{Nothing, FM}
    kwargs::KWARGS    
end

###### Fit 
function fit_transformer(X)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X; kwargs...)
    return
end  
function fit_transformer(X, weights::Jchemo.Weight)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, weights; kwargs...)
    return
end
function fit_predictor(X, Y)
    @show pnames(mod)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y; kwargs...)
    return
end  
function fit_predictor(X, Y, weights::Jchemo.Weight)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y, weights; kwargs...)
    return
end
###### Transform
function transf_mod(X; nlv = nothing)
    isnothing(nlv) ? transf(mod.fm, X) : 
        transf(mod.fm, X; nlv = nlv)
end
###### Predict 
function predict_mod(X; 
        nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        predict(mod.fm, X)
    elseif !isnothing(nlv) 
        predict(mod.fm, X; nlv = nlv)
    elseif !isnothing(lb) 
        predict(mod.fm, X; lb = lb)
    end
end 

