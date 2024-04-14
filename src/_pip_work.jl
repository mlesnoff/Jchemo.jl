###### Fit

function fit!(mod::Pipeline, X, Y = nothing)
    K = length(mod.mod)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    typ = [mod.mod[i].fun() for i in eachindex(mod.mod)]
    @inbounds for i = 1:(K - 1)
        if isa(typ[i], FunX)
            fit!(mod.mod[i], X)
        elseif isa(typ[i], FunXY)
            fit!(mod.mod[i], X, Y)
        end 
        X = transf(mod.mod[i], X)
    end
    if isa(typ[K], FunX)
        fit!(mod.mod[K], X)
    elseif isa(typ[K], FunXY)
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


