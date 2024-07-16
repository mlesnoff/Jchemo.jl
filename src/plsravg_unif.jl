function plsravg_unif(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plsravg_unif(X, Y, weights; kwargs...)
end

function plsravg_unif(X, Y, weights::Weight; kwargs...)
    plsravg_unif!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function plsravg_unif!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(Par, kwargs).par
    X = ensure_mat(X)
    n, p = size(X)
    nlv = (min(minimum(par.nlv), n, p):min(maximum(par.nlv), n, p))
    fm = plskern!(X, Y, weights; kwargs...)
    PlsravgUnif(fm, nlv)
end

function predict(object::PlsravgUnif, X)
    nlv = object.nlv
    le_nlv = length(nlv)
    zpred = predict(object.fm, X; nlv).pred
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


