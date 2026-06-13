## Define type 'JchemoModel' 

Base.@kwdef mutable struct JchemoModel{T <: Function, K <: Base.Pairs}
    algo::T   
    fitm
    kwargs::K
end

######## Fit

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

function fit!(model::JchemoModel, X, Y, weights::ProbabilityWeights)
    kwargs = values(model.kwargs)
    model.fitm = model.algo(X, Y, weights; kwargs...)
    return
end  

######## Transf

transf(model::JchemoModel, X) = transf(model.fitm, X)

transf(model::JchemoModel, X, nlv::Union{Int, AbstractVector{Int}}) = transf(model.fitm, X, nlv)

## 2-block

transf(model::JchemoModel, X, Y) = transf(model.fitm, X, Y)

transf(model::JchemoModel, X, Y, nlv::Union{Int, AbstractVector{Int}}) = transf(model.fitm, X, Y, nlv)

## Multiblock

transfbl(model::JchemoModel, X) = transfbl(model.fitm, X)

transfbl(model::JchemoModel, X, nlv::Union{Int, AbstractVector{Int}}) = transfbl(model.fitm, X, nlv)

transfbl(model::JchemoModel, X, Y) = transfbl(model.fitm, X, Y)

transfbl(model::JchemoModel, X, Y, nlv::Union{Int, AbstractVector{Int}}) = transfbl(model.fitm, X, Y, nlv)

######## Coef 

coef(model::JchemoModel) = coef(model.fitm)

function coef(model::JchemoModel, nlv::Union{Q, AbstractVector{Q}}) where Q <: Integer
    coef(model.fitm, nlv)
end

function coef(model::JchemoModel, lb::Union{Q, AbstractVector{Q}}) where Q <: AbstractFloat
    coef(model.fitm, lb)
end

######## Predict 

predict(model::JchemoModel, X) = predict(model.fitm, X)

function predict(model::JchemoModel, X, nlv::Union{Q, AbstractVector{Q}}) where Q <: Integer
    predict(model.fitm, X, nlv)
end

function predict(model::JchemoModel, X, lb::Union{Q, AbstractVector{Q}}) where Q <: AbstractFloat
    predict(model.fitm, X, lb)
end

######## Summary 

Base.summary(model::JchemoModel) = Base.summary(model.fitm)

Base.summary(model::JchemoModel, X) = Base.summary(model.fitm, X)

Base.summary(model::JchemoModel, X, Y) = Base.summary(model.fitm, X, Y)

