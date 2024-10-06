###### Fit

function fit!(model::Pipeline, X, Y = nothing)
    K = length(model.model)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    typ = [model.model[i].algo() for i in eachindex(model.model)]
    @inbounds for i = 1:(K - 1)
        if isa(typ[i], FunX)
            fit!(model.model[i], X)
        elseif isa(typ[i], FunXY)
            fit!(model.model[i], X, Y)
        end 
        X = transf(model.model[i], X)
    end
    if isa(typ[K], FunX)
        fit!(model.model[K], X)
    elseif isa(typ[K], FunXY)
        fit!(model.model[K], X, Y)
    end 
end

###### Transf

function transf(model::Pipeline, X)  
    K = length(model.model)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    @inbounds for i = 1:K
        X = transf(model.model[i], X)
    end
    X
end

###### Predict 

function predict(model::Pipeline, X)
    K = length(model.model)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    @inbounds for i = 1:(K - 1)
        X = transf(model.model[i], X)
    end
    predict(model.model[K], X)
end


