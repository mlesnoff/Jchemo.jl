function plsr_avg_cv(X, Y, weights = ones(size(X, 1)); nlv, 
        K = 2, typw = "bisquare", alpha = 0)
    plsr_avg_cv!(copy(X), copy(Y), weights; nlv = nlv, 
        K = K, typw = typw, alpha = alpha)
end

function plsr_avg_cv!(X, Y, weights = ones(size(X, 1)); nlv,
        K = 2, typw = "bisquare", alpha = 0)
    X = ensure_mat(X)
    n, p = size(X)
    nlv = eval(Meta.parse(nlv))
    nlv = (min(minimum(nlv), n, p):min(maximum(nlv), n, p))
    nlvmax = maximum(nlv)    
    segm = segmkf(n, K; rep = 1)
    res = gridcvlv(X, Y; segm = segm, score = rmsep, 
        fun = plskern, nlv = 0:nlvmax, verbose = false).res
    z = rowmean(rmcol(Matrix(res), 1))
    d = z .- findmin(z)[1]
    w = fweight(d, typw = typw, alpha = alpha)
    w .= mweight(w)
    fm = plskern!(X, Y, weights; nlv = nlvmax)
    PlsrAvgCri(fm, nlv, w)
end



