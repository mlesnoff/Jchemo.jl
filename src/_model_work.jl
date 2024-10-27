###### Fit

function fit!(model::JchemoModel, X)
    kwargs = values(model.kwargs)
    model.fitm = model.algo(X; kwargs...)
    return
end  

function fit!(model::JchemoModel, X, Y)
    kwargs = values(model.kwargs)
    model.fitm = model.algo(X, Y; kwargs...)
    return
end  
function fit!(model::JchemoModel, X, Y, weights::Weight)
    kwargs = values(model.kwargs)
    model.fitm = model.algo(X, Y, weights; kwargs...)
    return
end  

###### Transf

function transf(model::JchemoModel, X; nlv = nothing)
    isnothing(nlv) ? transf(model.fitm, X) : transf(model.fitm, X; nlv)
end
function transf(model::JchemoModel, X, Y; nlv = nothing)
    isnothing(nlv) ? transf(model.fitm, X, Y) : transf(model.fitm, X, Y; nlv)
end

function transfbl(model::JchemoModel, X; nlv = nothing)
    isnothing(nlv) ? transfbl(model.fitm, X) : transfbl(model.fitm, X; nlv)
end
function transfbl(model::JchemoModel, X, Y; nlv = nothing)
    isnothing(nlv) ? transfbl(model.fitm, X, Y) : transfbl(model.fitm, X, Y; nlv)
end

###### Predict 

function predict(model::JchemoModel, X; nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        predict(model.fitm, X)
    elseif !isnothing(nlv) 
        predict(model.fitm, X; nlv)
    elseif !isnothing(lb) 
        predict(model.fitm, X; lb)
    end
end  

###### Coef 

function coef(model::JchemoModel; nlv = nothing, lb = nothing)
    if isnothing(nlv) && isnothing(lb)
        coef(model.fitm)
    elseif !isnothing(nlv) 
        coef(model.fitm; nlv)
    elseif !isnothing(lb) 
        coef(model.fitm; lb)
    end
end

###### Summary 

function Base.summary(model::JchemoModel)
    Base.summary(model.fitm)
end
function Base.summary(model::JchemoModel, X)
    Base.summary(model.fitm, X)
end
function Base.summary(model::JchemoModel, X, Y)
    Base.summary(model.fitm, X, Y)
end

