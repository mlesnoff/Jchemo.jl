struct PlsravgShenk
    fm::Plsr
    nlv
end

function plsravg_shenk(X, Y, weights = ones(nro(X)); nlv, 
        scal::Bool = false)
    plsravg_shenk!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv, 
        scal = scal)
end

function plsravg_shenk!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv, 
        scal::Bool = false)
    n, p = size(X)
    nlv = eval(Meta.parse(nlv))
    nlv = (min(minimum(nlv), n, p):min(maximum(nlv), n, p))
    nlvmax = maximum(nlv)
    nlv = max(minimum(nlv), 1):nlvmax       
    fm = plskern!(X, Y, weights; nlv = nlvmax, scal = scal)
    PlsravgShenk(fm, nlv)
end

function predict(object::PlsravgShenk, X)
    X = ensure_mat(X)
    m = nro(X)
    q = nro(object.fm.C)
    nlv = object.nlv
    le_nlv = length(nlv)
    zpred = predict(object.fm, X; nlv = nlv).pred
    if(le_nlv == 1)
        pred = zpred
    else
        pred = similar(X, m, q)
        w = wshenk(object.fm, X).w[:, nlv]
        for i = 1:m
            zw = mweight(vrow(w, i))
            acc = zw[1] * vrow(zpred[1], i)
            @inbounds for j = 2:le_nlv
                acc .+= zw[j] * vrow(zpred[j], i)
            end
            pred[i, :] .= acc
        end
    end
    (pred = pred, predlv = zpred, w = w)
end


