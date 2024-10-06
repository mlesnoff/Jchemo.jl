###### Fit

function fit!(model::Jchemo.JchemoModel, X)
    kwargs = values(model.kwargs)
    model.fm = model.algo(X; kwargs...)
    return
end  

function fit!(model::Jchemo.JchemoModel, X, Y)
    kwargs = values(model.kwargs)
    model.fm = model.algo(X, Y; kwargs...)
    return
end  
function fit!(model::Jchemo.JchemoModel, X, Y, weights::Weight)
    kwargs = values(model.kwargs)
    model.fm = model.algo(X, Y, weights; kwargs...)
    return
end  

###### Transf

function transf(model::Jchemo.JchemoModel, X; nlv = nothing)
    isnothing(nlv) ? transf(model.fm, X) : transf(model.fm, X; nlv)
end
function transf(model::Jchemo.JchemoModel, X, Y; nlv = nothing)
    isnothing(nlv) ? transf(model.fm, X, Y) : transf(model.fm, X, Y; nlv)
end

function transfbl(model::Jchemo.JchemoModel, X; nlv = nothing)
    isnothing(nlv) ? transfbl(model.fm, X) : transfbl(model.fm, X; nlv)
end
function transfbl(model::Jchemo.JchemoModel, X, Y; nlv = nothing)
    isnothing(nlv) ? transfbl(model.fm, X, Y) : transfbl(model.fm, X, Y; nlv)
end

###### Predict 

function predict(model::Jchemo.JchemoModel, X; nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        predict(model.fm, X)
    elseif !isnothing(nlv) 
        predict(model.fm, X; nlv)
    elseif !isnothing(lb) 
        predict(model.fm, X; lb)
    end
end  

###### Coef 

function coef(model::Jchemo.JchemoModel; nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        coef(model.fm)
    elseif !isnothing(nlv) 
        coef(model.fm; nlv)
    elseif !isnothing(lb) 
        coef(model.fm; lb)
    end
end

###### Summary 

function Base.summary(model::Jchemo.JchemoModel)
    Base.summary(model.fm)
end
function Base.summary(model::Jchemo.JchemoModel, X)
    Base.summary(model.fm, X)
end
function Base.summary(model::Jchemo.JchemoModel, X, Y)
    Base.summary(model.fm, X, Y)
end

