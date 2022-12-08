struct PlsravgEns
    fm
    w::Vector{Float64}
end

function plsravg_ens(Xbl, Y, weights = ones(size(Xbl[1], 1)); 
        nlv, typw = "unif")
    nbl = length(Xbl)
    fm = list(nbl)
    w = ones(nbl)
    Threads.@threads for k = 1:nbl
        fm[k] = plsravg(Xbl[k], Y, weights; nlv = nlv)
        if typw != "unif"
            zpred = vec(predict(fm[k], Xbl[k]).pred)
            w[k] = sum(1 - cor(Y, zpred).^2)
        end
    end
    if typw != "unif"
        w .= fweight(w; typw = typw)
    end
    w .= mweight(w)
    PlsravgEns(fm, w)
end

#"""
#    predict(object::PlsravgEns, Xbl)
#Compute Y-predictions from a fitted model.
#* `object` : The fitted model.
#* `Xbl` : A list (vector) of X-data for which predictions are computed.
#""" 
function predict(object::PlsravgEns, Xbl)
    nbl = length(object.fm)
    pred = object.w[1] * predict(object.fm[1], Xbl[1]).pred
    if nbl > 1
        @inbounds for k = 2:nbl
            pred .+= object.w[k] * predict(object.fm[k], Xbl[k]).pred
        end
    end 
    (pred = pred,)
end

