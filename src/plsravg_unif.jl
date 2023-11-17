function plsravg_unif(X, Y; par = Par())
    weights = mweight(ones(eltype(X[1, 1]), nro(X)))
    plsravg_unif(X, Y, weights; par)
end

function plsravg_unif(X, Y, weights::Weight; par = Par())
    plsravg_unif!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; par)
end

function plsravg_unif!(X::Matrix, Y::Matrix, weights::Weight; 
        par = Par())
    X = ensure_mat(X)
    n, p = size(X)
    nlv = (min(minimum(par.nlv), n, p):min(maximum(par.nlv), n, p))
    nlvmax = maximum(nlv)
    par.nlv = nlvmax     
    fm = plskern!(X, Y, weights; par)
    PlsravgUnif(fm, nlv)
end

function predict(object::PlsravgUnif, X)
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


