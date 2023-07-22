function pcanipals(X, weights = ones(nro(X)); nlv, 
    gs = true, niter = 100, scal::Bool = false)
    pcanipals!(copy(ensure_mat(X)), weights; nlv = nlv, 
        gs = gs, niter = niterscal = scal)
end

function pcanipals!(X::Matrix, weights = ones(nro(X)); nlv, 
    gs = true, niter = 100, scal::Bool = false)
    n, p = size(X)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    xscales = ones(p)
    if scal 
        xscales .= colstd(X, weights)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end
    sqrtw = sqrt.(weights)
    X .= Diagonal(sqrtw) * X
    for a = 1:nlv
        res = nipals(X; tol = tol, maxit = maxit)
        P[:, a] .= res.u           
        T[:, a] .= t    




    res = eigen!(Symmetric(X' * X); sortby = x -> -abs(x)) 
    P = res.vectors[:, 1:nlv]
    eig = res.values[1:min(n, p)]
    eig[eig .< 0] .= 0
    sv = sqrt.(eig)
    T = Diagonal(1 ./ sqrtw) * X * P
    
    Pca(T, P, sv, xmeans, xscales, weights, nothing, nothing) 
end

