######---- Fit
## Transformers
function fit!(mod::Jchemo.Transformer, X)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X; kwargs...)
    return
end  
function fit!(mod::Jchemo.Transformer, X, weights::Weight)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, weights; kwargs...)
    return
end
## Transformers 2 blocks
function fit!(mod::Jchemo.Transformer, X, Y)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y; kwargs...)
    return
end  
function fit!(mod::Jchemo.Transformer, X, Y, weights::Weight)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y, weights; kwargs...)
    return
end  
## Predictors
function fit!(mod::Jchemo.Predictor, X, Y)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y; kwargs...)
    return
end  
function fit!(mod::Jchemo.Predictor, X, Y, weights::Weight)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X, Y, weights; kwargs...)
    return
end 
function fit!(mod::PredictorNoY, X)
    kwargs = values(mod.kwargs)
    mod.fm = mod.fun(X; kwargs...)
    return
end   

######---- Transf
function transf(mod::Union{Jchemo.Transformer, Jchemo.Predictor}, X; nlv = nothing)
    isnothing(nlv) ? transf(mod.fm, X) : 
        transf(mod.fm, X; nlv)
end
## X, Y
function transf(mod::Union{Jchemo.Transformer, Jchemo.Predictor}, X, Y; nlv = nothing)
    isnothing(nlv) ? transf(mod.fm, X, Y) : 
        transf(mod.fm, X, Y; nlv)
end

######---- Transfbl
function transfbl(mod::Union{Jchemo.Transformer, Jchemo.Predictor}, X; nlv = nothing)
    isnothing(nlv) ? transfbl(mod.fm, X) : 
        transfbl(mod.fm, X; nlv)
end
## X, Y
function transfbl(mod::Union{Jchemo.Transformer, Jchemo.Predictor}, X, Y; nlv = nothing)
    isnothing(nlv) ? transfbl(mod.fm, X, Y) : 
    transfbl(mod.fm, X, Y; nlv)
end

######---- Predict 
function predict(mod::Union{Jchemo.Transformer, Jchemo.Predictor}, X; nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        predict(mod.fm, X)
    elseif !isnothing(nlv) 
        predict(mod.fm, X; nlv)
    elseif !isnothing(lb) 
        predict(mod.fm, X; lb)
    end
end  
predict(mod::PredictorNoY, X) = predict(mod.fm, X)

######---- Coef 
function coef(mod::Jchemo.Predictor; nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        coef(mod.fm)
    elseif !isnothing(nlv) 
        coef(mod.fm; nlv)
    elseif !isnothing(lb) 
        coef(mod.fm; lb)
    end
end

######---- Summary 
function Base.summary(mod::Union{Jchemo.Transformer, Jchemo.Predictor})
    Base.summary(mod.fm)
end
function Base.summary(mod::Union{Jchemo.Transformer, Jchemo.Predictor}, X)
    Base.summary(mod.fm, X)
end
function Base.summary(mod::Jchemo.Transformer, X, Y)
    Base.summary(mod.fm, X, Y)
end