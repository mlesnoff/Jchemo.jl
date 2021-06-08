struct PlsrAgg
    fm::Plsr
    nlv
end

"""
    plsr_agg!(X, Y, weights = ones(size(X, 1)) ; nlv)
""" 
function plsr_agg!(X, Y, weights = ones(size(X, 1)) ; nlv)
    n = size(X, 1)
    p = size(X, 2)
    nlv = eval(Meta.parse(nlv))
    nlv = (max(minimum(nlv), 0):min(maximum(nlv), n, p))
    fm = plskern!(X, Y, weights; nlv = maximum(nlv))
    PlsrAgg(fm, nlv)
end

"""
    plsr_agg(X, Y, weights = ones(size(X, 1)) ; nlv)
""" 
function plsr_agg(X, Y, weights = ones(size(X, 1)) ; nlv)
    res = plsr_agg!(copy(X), copy(Y), weights; nlv)
    res
end

"""
predict(object::PlsrAgg, X)
"""
function predict(object::PlsrAgg, X)
    nlv = object.nlv
    le_nlv = length(nlv)
    zpred = predict(object.fm, X; nlv = nlv).pred
    if(le_nlv == 1)
        pred = zpred
    else
        acc = zpred[1]
        for i = 2:le_nlv
            acc .+= zpred[i]
        end
        pred = acc ./ le_nlv
    end
    (pred = pred, predlv = zpred)
end


