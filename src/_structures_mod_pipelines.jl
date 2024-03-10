struct Pipeline
    mod::Tuple
end
pip(args...) = Pipeline(values(args))

###### Fit
function fit!(mod::Pipeline, X, Y = nothing)
    K = length(mod.mod)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    @inbounds for i = 1:(K - 1)
        if isa(mod.mod[i], Transformer)
            fit!(mod.mod[i], X)
        elseif isa(mod.mod[i], Predictor)
            fit!(mod.mod[i], X, Y)
        end
        X = transf(mod.mod[i], X)
    end
    if isa(mod.mod[K], Transformer)
        fit!(mod.mod[K], X)
    elseif isa(mod.mod[K], Predictor)
        fit!(mod.mod[K], X, Y)
    end
end

###### Transf
function transf(mod::Pipeline, X)  
    K = length(mod.mod)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    @inbounds for i = 1:K
        X = transf(mod.mod[i], X)
    end
    X
end

###### Predict 
function predict(mod::Pipeline, X)
    K = length(mod.mod)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    @inbounds for i = 1:(K - 1)
        X = transf(mod.mod[i], X)
    end
    predict(mod.mod[K], X)
end


