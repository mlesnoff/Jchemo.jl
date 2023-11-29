function plsravg_unif(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plsravg_unif(X, Y, weights; values(kwargs)...)
end

function plsravg_unif(X, Y, weights::Weight; kwargs...)
    plsravg_unif!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; values(kwargs)...)
end

function plsravg_unif!(X::Matrix, Y::Matrix, weights::Weight; 
        par = Par())
    X = ensure_mat(X)
    n, p = size(X)
    nlv = (min(minimum(par.nlv), n, p):min(maximum(par.nlv), n, p))
    nlvmax = maximum(nlv)
    par.nlv = nlvmax     
    fm = plskern!(X, Y, weights; values(kwargs)...)
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


