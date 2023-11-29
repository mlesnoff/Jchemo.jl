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

function transf(mod::Union{Transformer, Predictor}, X)
    transf(mod.fm, X)
end  

#### Functions

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

#### Pipelines

function fit!(mod::Tuple, X)
    K = length(mod)
    for i = 1:(K - 1)
        fit!(mod[i], X)
        X = transf(mod[i], X)
    end
    fit!(mod[K], X)
end

function fit!(mod::Tuple, X, Y)
    K = length(mod)
    for i = 1:(K - 1)
        fit!(mod[i], X, Y)
        X = transf(mod[i], X)
    end
    fit!(mod[K], X, Y)
end

## Future: can build 
## fit!(mod::Tuple, X, weights::Weight)
## fit!(mod::Tuple, X, Y, weights::Weight)

#function fit!(mod::Tuple, X, Y = nothing)
#    K = length(mod)
#    for i = 1:(K - 1)
#        if isa(mod[i], Transformer)
#            fit!(mod[i], X)
#        elseif isa(mod[i], Predictor)
#            fit!(mod[i], X, Y)
#        end
#        X = transf(mod[i], X)
#    end
#    if isa(mod[K], Transformer)
#        fit!(mod[K], X)
#    elseif isa(mod[K], Predictor)
#        fit!(mod[K], X, Y)
#    end
#end

function transf(mod::Tuple, X)  
    K = length(mod)
    for i = 1:K
        X = transf(mod[i], X)
    end
    X
end




