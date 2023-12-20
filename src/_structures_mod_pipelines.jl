function fit!(mod::Tuple, X)
    K = length(mod)
    for i = 1:(K - 1)
        fit!(mod[i], X)
        X = transf(mod[i], X)
    end
    fit!(mod[K], X)
end

function fit!(mod::Tuple, X, Y)
    K = length(mod)
    for i = 1:(K - 1)
        fit!(mod[i], X, Y)
        X = transf(mod[i], X)
    end
    fit!(mod[K], X, Y)
end
## In the future, build: 
## fit!(mod::Tuple, X, weights::Weight)
## fit!(mod::Tuple, X, Y, weights::Weight)

function transf(mod::Tuple, X)  
    K = length(mod)
    for i = 1:K
        X = transf(mod[i], X)
    end
    X
end


#function fit!(mod::Tuple, X, Y = nothing)
#    K = length(mod)
#    for i = 1:(K - 1)
#        if isa(mod[i], Transformer)
#            fit!(mod[i], X)
#        elseif isa(mod[i], Predictor)
#            fit!(mod[i], X, Y)
#        end
#        X = transf(mod[i], X)
#    end
#    if isa(mod[K], Transformer)
#        fit!(mod[K], X)
#    elseif isa(mod[K], Predictor)
#        fit!(mod[K], X, Y)
#    end
#end

# predict : see old_new in \Jchemo2
