###### Fit
function fit!(mod::Transformer, X)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X; kwargs...)
    return
end  
function fit!(mod::Transformer, X, 
        weights::Weight)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, weights; kwargs...)
    return
end  
function fit!(mod::Predictor, X, Y)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y; kwargs...)
    return
end  
function fit!(mod::Predictor, X, Y, 
        weights::Weight)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y, weights; kwargs...)
    return
end 
function fit!(mod::PredictorNoY, X)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X; kwargs...)
    return
end   
## 2 blocks
function fit!(mod::Transformer, X, Y)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y; kwargs...)
    return
end  
function fit!(mod::Transformer, X, Y, 
        weights::Weight)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y, weights; kwargs...)
    return
end  
## >= 2 blocks 
function fit!(mod::Transformer, Xbl::Vector{Matrix})
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(Xbl; kwargs...)
    return
end  
function fit!(mod::Transformer, Xbl::Vector{Matrix}, 
        weights::Weight)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(Xbl, weights; kwargs...)
    return
end  

###### Transf
function transf(mod::Union{Transformer, Predictor}, 
        X; nlv = nothing)
    isnothing(nlv) ? transf(mod.fm, X) : 
        transf(mod.fm, X; nlv)
end
## 2 blocks
function transf(mod::Union{Transformer, Predictor}, 
        X, Y; nlv = nothing)
    isnothing(nlv) ? transf(mod.fm, X, Y) : 
        transf(mod.fm, X, Y; nlv)
end
## >= 2 blocks 
function transf(mod::Union{Transformer, Predictor}, 
        Xbl::Vector{Matrix}; nlv = nothing)
    isnothing(nlv) ? transf(mod.fm, Xbl) : 
        transf(mod.fm, Xbl; nlv)
end

###### Transfbl
## 2 blocks
function transfbl(mod::Union{Transformer, Predictor}, 
    X, Y; nlv = nothing)
isnothing(nlv) ? transfbl(mod.fm, X, Y) : 
    transfbl(mod.fm, X, Y; nlv)
end
## >= 2 blocks 
function transfbl(mod::Union{Transformer, Predictor}, 
    Xbl::Vector{Matrix}; nlv = nothing)
    isnothing(nlv) ? transfbl(mod.fm, Xbl) : 
        transfbl(mod.fm, Xbl; nlv)
end

###### Predict 
function predict(mod::Union{Transformer, Predictor}, 
        X; nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        predict(mod.fm, X)
    elseif !isnothing(nlv) 
        predict(mod.fm, X; nlv)
    elseif !isnothing(lb) 
        predict(mod.fm, X; lb)
    end
end  
predict(mod::PredictorNoY, X) = predict(mod.fm, X)

###### Coef 
function coef(mod::Jchemo.Predictor; 
        nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        coef(mod.fm)
    elseif !isnothing(nlv) 
        coef(mod.fm; nlv)
    elseif !isnothing(lb) 
        coef(mod.fm; lb)
    end
end

###### Summary 
function Base.summary(mod::Union{Jchemo.Transformer, 
        Jchemo.Predictor})
    Base.summary(mod.fm)
end
function Base.summary(mod::Union{Jchemo.Transformer, 
        Jchemo.Predictor}, X)
    Base.summary(mod.fm, X)
end
