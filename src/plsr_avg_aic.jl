struct PlsrAvgCri
    fm::Plsr
    nlv
    w::Vector
end

struct PlsrStack
    fm::Plsr
    nlv
    w::Vector
    Xstack  # = View
    ystack::Array
    weights_stack::Array
end

function plsr_avg_aic(X, y, weights = ones(size(X, 1)); nlv, 
        bic = false, typw = "bisquare", alpha = 0)
    plsr_avg_aic!(copy(X), copy(y), weights; nlv = nlv, bic = bic, 
        typw = typw, alpha = alpha)
end

function plsr_avg_aic!(X, y, weights = ones(size(X, 1)); nlv,
        bic = false, typw = "bisquare", alpha = 0)
    X = ensure_mat(X)
    n, p = size(X)
    nlv = eval(Meta.parse(nlv))
    nlv = (min(minimum(nlv), n, p):min(maximum(nlv), n, p))
    nlvmax = maximum(nlv)    
    res = aicplsr(X, y; nlv = nlvmax, bic = bic)
    d = res.delta.aic[nlv .+ 1]
    #d = res.crit.aic[nlv .+ 1]
    w = fweight(d, typw = typw, alpha = alpha)
    w .= mweight(w)
    fm = plskern!(X, y, weights; nlv = nlvmax)
    PlsrAvgCri(fm, nlv, w)
end

function predict(object::Union{PlsrAvgCri, PlsrStack}, X)
    nlv = object.nlv
    le_nlv = length(nlv)
    zpred = predict(object.fm, X; nlv = nlv).pred
    if(le_nlv == 1)
        pred = zpred
    else
        acc = object.w[1] * copy(zpred[1])
        @inbounds for i = 2:le_nlv
            acc .+= object.w[i] * zpred[i]
        end
        pred = acc
    end
    (pred = pred, predlv = zpred)
end


