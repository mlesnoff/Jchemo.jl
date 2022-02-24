struct PlsrAvgUnif
    fm::Plsr
    nlv
end

function plsr_avg_unif(X, Y, weights = ones(size(X, 1)); nlv)
    plsr_avg_unif!(copy(X), copy(Y), weights; nlv = nlv)
end

function plsr_avg_unif!(X, Y, weights = ones(size(X, 1)); nlv)
    X = ensure_mat(X)
    n, p = size(X)
    nlv = eval(Meta.parse(nlv))
    nlv = (min(minimum(nlv), n, p):min(maximum(nlv), n, p))
    nlvmax = maximum(nlv)     
    fm = plskern!(X, Y, weights; nlv = nlvmax)
    PlsrAvgUnif(fm, nlv)
end

function predict(object::PlsrAvgUnif, X)
    nlv = object.nlv
    le_nlv = length(nlv)
    zpred = predict(object.fm, X; nlv = nlv).pred
    if(le_nlv == 1)
        pred = zpred
    else
        acc = copy(zpred[1])
        @inbounds for i = 2:le_nlv
            acc .+= zpred[i]
        end
        pred = acc / le_nlv
    end
    (pred = pred, predlv = zpred)
end


