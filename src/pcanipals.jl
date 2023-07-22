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
    t = similar(X, n)
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    sv = similar(X, nlv)
    niter = list(nlv, Int64)
    for a = 1:nlv
        res = nipals(X; tol = tol, maxit = maxit)
        t .= res.u * res.sv
        sv[a] = res.sv
        P[:, a] .= res.v           
        T[:, a] .= t
        X .-= t * res.v'
        t ./= sqrtw
        niter[a] = res.niter
    end    
    #T = Diagonal(1 ./ sqrtw) * X * P
    Pca(T, P, sv, xmeans, xscales, weights, niter, nothing) 
end

