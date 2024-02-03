######---- Fit
## Transformers
function fit!(mo::Jchemo.Transformer, X)
    kwargs = values(mo.kwargs)
    mo.fm = mo.fun(X; kwargs...)
    return
end  
function fit!(mo::Jchemo.Transformer, X, 
        weights::Weight)
    kwargs = values(mo.kwargs)
    mo.fm = mo.fun(X, weights; kwargs...)
    return
end
## Transformers 2 blocks
function fit!(mo::Jchemo.Transformer, X, Y)
    kwargs = values(mo.kwargs)
    mo.fm = mo.fun(X, Y; kwargs...)
    return
end  
function fit!(mo::Jchemo.Transformer, X, Y, 
        weights::Weight)
    kwargs = values(mo.kwargs)
    mo.fm = mo.fun(X, Y, weights; kwargs...)
    return
end  
## Predictors
function fit!(mo::Jchemo.Predictor, X, Y)
    kwargs = values(mo.kwargs)
    mo.fm = mo.fun(X, Y; kwargs...)
    return
end  
function fit!(mo::Jchemo.Predictor, X, Y, 
        weights::Weight)
    kwargs = values(mo.kwargs)
    mo.fm = mo.fun(X, Y, weights; kwargs...)
    return
end 
function fit!(mo::PredictorNoY, X)
    kwargs = values(mo.kwargs)
    mo.fm = mo.fun(X; kwargs...)
    return
end   

######---- Transf
function transf(mo::Union{Jchemo.Transformer, Jchemo.Predictor}, 
        X; nlv = nothing)
    isnothing(nlv) ? transf(mo.fm, X) : 
        transf(mo.fm, X; nlv)
end
## X, Y
function transf(mo::Union{Jchemo.Transformer, Jchemo.Predictor}, 
        X, Y; nlv = nothing)
    isnothing(nlv) ? transf(mo.fm, X, Y) : 
        transf(mo.fm, X, Y; nlv)
end

######---- Transfbl
function transfbl(mo::Union{Jchemo.Transformer, Jchemo.Predictor}, 
        X; nlv = nothing)
    isnothing(nlv) ? transfbl(mo.fm, X) : 
        transfbl(mo.fm, X; nlv)
end
## X, Y
function transfbl(mo::Union{Jchemo.Transformer, Jchemo.Predictor}, 
        X, Y; nlv = nothing)
    isnothing(nlv) ? transfbl(mo.fm, X, Y) : 
    transfbl(mo.fm, X, Y; nlv)
end

######---- Predict 
function predict(mo::Union{Jchemo.Transformer, Jchemo.Predictor}, 
        X; nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        predict(mo.fm, X)
    elseif !isnothing(nlv) 
        predict(mo.fm, X; nlv)
    elseif !isnothing(lb) 
        predict(mo.fm, X; lb)
    end
end  
predict(mo::PredictorNoY, X) = predict(mo.fm, X)

######---- Coef 
function coef(mo::Jchemo.Predictor; 
        nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        coef(mo.fm)
    elseif !isnothing(nlv) 
        coef(mo.fm; nlv)
    elseif !isnothing(lb) 
        coef(mo.fm; lb)
    end
end

######---- Summary 
function Base.summary(mo::Union{Jchemo.Transformer, Jchemo.Predictor})
    Base.summary(mo.fm)
end
function Base.summary(mo::Union{Jchemo.Transformer, Jchemo.Predictor}, X)
    Base.summary(mo.fm, X)
end
function Base.summary(mo::Jchemo.Transformer, X, Y)
    Base.summary(mo.fm, X, Y)
end