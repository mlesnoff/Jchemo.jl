struct Pipeline
    mod::Tuple
end
pip(args...) = Pipeline(values(args))

###### Fit
function fit!(mod::Pipeline, X, Y = nothing)
    K = length(mod.mod)
    for i = 1:(K - 1)
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
    for i = 1:K
        X = transf(mod.mod[i], X)
    end
    X
end

###### Predict 
function predict(mod::Pipeline, X)
    K = length(mod.mod)
    for i = 1:(K - 1)
        X = transf(mod.mod[i], X)
    end
    predict(mod.mod[K], X)
end


#function fit!(mod::Tuple, X)
#    K = length(mod)
#    for i = 1:(K - 1)
#        fit!(mod[i], X)
#        X = transf(mod[i], X)
#    end
#    fit!(mod[K], X)
#end

#function fit!(mod::Tuple, X, Y)
#    K = length(mod)
#    for i = 1:(K - 1)
#        fit!(mod[i], X, Y)
#        X = transf(mod[i], X)
#    end
#    fit!(mod[K], X, Y)
#end
## In the future, build: 
## fit!(mod::Tuple, X, weights::Weight)
## fit!(mod::Tuple, X, Y, weights::Weight)
