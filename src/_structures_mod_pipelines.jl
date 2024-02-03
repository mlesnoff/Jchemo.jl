struct Pipeline
    mo::Tuple
end
pip(args...) = Pipeline(values(args))

###### Fit
function fit!(mo::Pipeline, X, Y = nothing)
    K = length(mo.mo)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    for i = 1:(K - 1)
        if isa(mo.mo[i], Transformer)
            fit!(mo.mo[i], X)
        elseif isa(mo.mo[i], Predictor)
            fit!(mo.mo[i], X, Y)
        end
        X = transf(mo.mo[i], X)
    end
    if isa(mo.mo[K], Transformer)
        fit!(mo.mo[K], X)
    elseif isa(mo.mo[K], Predictor)
        fit!(mo.mo[K], X, Y)
    end
end

###### Transf
function transf(mo::Pipeline, X)  
    K = length(mo.mo)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    for i = 1:K
        X = transf(mo.mo[i], X)
    end
    X
end

###### Predict 
function predict(mo::Pipeline, X)
    K = length(mo.mo)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    for i = 1:(K - 1)
        X = transf(mo.mo[i], X)
    end
    predict(mo.mo[K], X)
end


