function plsr_avg_cv(X, Y, weights = ones(nro(X)); nlv, 
        K = 2, typw = "bisquare", 
        alpha = 0, scal = false)
    plsr_avg_cv!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv, 
        K = K, typw = typw,
        alpha = alpha, scal = scal)
end

function plsr_avg_cv!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        K = 2, typw = "bisquare",
        alpha = 0, scal = false)
    n, p = size(X)
    nlv = eval(Meta.parse(nlv))
    nlv = (min(minimum(nlv), n, p):min(maximum(nlv), n, p))
    nlvmax = maximum(nlv)    
    segm = segmkf(n, K; rep = 1)
    pars = mpar(scal = scal)
    res = gridcvlv(X, Y; segm = segm, score = rmsep, 
        fun = plskern, nlv = 0:nlvmax, pars = pars, verbose = false).res
    z = rowmean(rmcol(Matrix(res), 1))
    d = (z .- findmin(z)[1])[nlv .+ 1]
    w = fweight(d, typw = typw, alpha = alpha)
    w .= mweight(w)
    fm = plskern!(X, Y, weights; nlv = nlvmax, scal = scal)
    PlsrAvgCri(fm, nlv, w)
end



