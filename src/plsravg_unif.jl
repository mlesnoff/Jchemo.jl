struct PlsravgUnif
    fm::Plsr
    nlv
end

function plsravg_unif(X, Y, weights = ones(nro(X)); nlv,
        scal = false)
    plsravg_unif!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv,
        scal = scal)
end

function plsravg_unif!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        scal = false)
    X = ensure_mat(X)
    n, p = size(X)
    nlv = eval(Meta.parse(nlv))
    nlv = (min(minimum(nlv), n, p):min(maximum(nlv), n, p))
    nlvmax = maximum(nlv)     
    fm = plskern!(X, Y, weights; nlv = nlvmax, scal = scal)
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


