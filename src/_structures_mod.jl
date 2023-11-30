Base.@kwdef mutable struct Transformer{FUN <: Function, FM, KWARGS <: Base.Pairs}
    fun::FUN   
    fm::Union{Nothing, FM}
    kwargs::KWARGS    
end

Base.@kwdef mutable struct Predictor{FUN <: Function, FM, KWARGS <: Base.Pairs}
    fun::FUN   
    fm::Union{Nothing, FM}
    kwargs::KWARGS    
end

#### Fit

function fit!(mod::Transformer, X)
    mod.fm = mod.fun(X; values(mod.kwargs)...)
    return
end  
function fit!(mod::Transformer, X, weights::Weight)
    mod.fm = mod.fun(X, weights; values(mod.kwargs)...)
    return
end  

function fit!(mod::Predictor, X, Y)
    mod.fm = mod.fun(X, Y; values(mod.kwargs)...)
end  
function fit!(mod::Predictor, X, Y, weights::Weight)
    mod.fm = mod.fun(X, Y, weights; values(mod.kwargs)...)
end  

#### Transf

function transf(mod::Union{Transformer, Predictor}, X; 
        nlv = nothing)
    isnothing(nlv) ? transf(mod.fm, X) : transf(mod.fm, X; nlv = nlv)
end  

#### Predict 

function predict(mod::Union{Transformer, Predictor}, X; 
        nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        predict(mod.fm, X)
    elseif !isnothing(nlv) 
        predict(mod.fm, X; nlv = nlv)
    elseif !isnothing(lb) 
        predct(mod.fm, X; lb = lb)
    end
end  

#### Functions building mod

detrend(; kwargs...) = Transformer{Function, Detrend, Base.Pairs}(detrend, nothing, kwargs)
fdif(; kwargs...) = Transformer{Function, Fdif, Base.Pairs}(fdif, nothing, kwargs)
interpl(; kwargs...) = Transformer{Function, Interpl, Base.Pairs}(interpl, nothing, kwargs)
mavg(; kwargs...) = Transformer{Function, Mavg, Base.Pairs}(mavg, nothing, kwargs)
savgol(; kwargs...) = Transformer{Function, Savgol, Base.Pairs}(savgol, nothing, kwargs)
snv(; kwargs...) = Transformer{Function, Snv, Base.Pairs}(snv, nothing, kwargs)

center(; kwargs...) = Transformer{Function, Center, Base.Pairs}(center, nothing, kwargs)
scale(; kwargs...) = Transformer{Function, Scale, Base.Pairs}(scale, nothing, kwargs)
cscale(; kwargs...) = Transformer{Function, Cscale, Base.Pairs}(cscale, nothing, kwargs)

plskern(; kwargs...) = Predictor{Function, Plsr, Base.Pairs}(plskern, nothing, kwargs)


