###### Fit

function fit!(mod::Pipeline, X, Y = nothing)
    K = length(mod.mod)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    typ = [mod.mod[i].typ for i in eachindex(mod.mod)]
    @inbounds for i = 1:(K - 1)
        isequal(typ[i], :x) ? fit!(mod.mod[i], X) : fit!(mod.mod[i], X, Y) 
        X = transf(mod.mod[i], X)
    end
    isequal(typ[K], :x) ? fit!(mod.mod[K], X) : fit!(mod.mod[K], X, Y) 
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


