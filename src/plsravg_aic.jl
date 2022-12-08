struct PlsravgCri
    fm::Plsr
    nlv
    w::Vector
end

struct Plsrstack
    fm::Plsr
    nlv
    w::Vector
    Xstack  # = View
    ystack::Array
    weightsstack::Array
end

function plsravg_aic(X, y, weights = ones(nro(X)); nlv, 
        bic = false, typw = "bisquare",
        alpha = 0, scal = false)
    plsravg_aic!(copy(ensure_mat(X)), copy(ensure_mat(y)), weights; nlv = nlv,
        bic = bic, typw = typw, 
        alpha = alpha, scal = scal)
end

function plsravg_aic!(X::Matrix, y::Matrix, weights = ones(nro(X)); nlv,
        bic = false, typw = "bisquare", 
        alpha = 0, scal = false)
    n, p = size(X)
    nlv = eval(Meta.parse(nlv))
    nlv = (min(minimum(nlv), n, p):min(maximum(nlv), n, p))
    nlvmax = maximum(nlv)
    res = aicplsr(X, y; nlv = nlvmax, bic = bic, scal = scal)
    # To not break lwplsravg_aic when there are NaN in delta.Aic
    z = res.delta.aic
    l = length(z[isnan.(z)])
    if l > 0
        z[isnan.(z)] .= mean(z[isnan.(z) .== 0]) .+ 1e-8 * sort(abs.(rand(l)))
    end  
    # End
    d = z[nlv .+ 1]
    w = fweight(d, typw = typw, alpha = alpha)
    w .= mweight(w)
    fm = plskern!(X, y, weights; nlv = nlvmax, scal = scal)
    PlsravgCri(fm, nlv, w)
end

function predict(object::Union{PlsravgCri, Plsrstack}, X)
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


