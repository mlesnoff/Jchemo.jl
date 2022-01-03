struct MbplsrAvg
    fm
    w::Vector{Float64}
end

function mbplsr_avg(X, Y, weights = ones(size(X[1], 1)); nlv, w = nothing)
    nbl = length(X)
    isnothing(w) ? w = ones(nbl) / nbl : nothing
    fm = list(nbl)
    @inbounds for i = 1:nbl
        fm[i] = plskern(X[i], Y, weights; nlv = nlv[i])
    end
    MbplsrAvg(fm, w)
end

"""
    predict(object::MbplsrAvg, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : A list (vector) of X-data for which predictions are computed.
""" 
function predict(object::MbplsrAvg, X)
    nbl = length(object.fm)
    pred = object.w[1] * predict(object.fm[1], X[1]).pred
    if nbl > 1
        @inbounds for i = 2:nbl
            pred .+= object.w[i] * predict(object.fm[i], X[i]).pred
        end
    end 
    (pred = pred,)
end

