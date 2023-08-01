function spca(X, weights = ones(nro(X)); nlv,
        nvar = nco(X),
        delta = 0, gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
    spca!(copy(ensure_mat(X)), weights; nlv = nlv,
        nvar = nvar, 
        delta = delta, gs = gs, tol = tol, maxit = maxit, 
        scal = scal)
end

function spca!(X::Matrix, weights = ones(nro(X)); nlv, 
        nvar = nco(X),
        delta = 0, gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
    n, p = size(X)
    nlv = min(nlv, n, p)
    length(nvar) == 1 ? nvar = repeat([nvar], nlv) : nothing
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
    sellv = list(nlv, Vector{Int64})
    for a = 1:nlv
        res = snipals(X; 
            delta = delta, tol = tol, maxit = maxit)
        t .= res.u * res.sv
        tt = dot(t, t)
        X .-= t * t' * X / tt        
        T[:, a] .= t ./ sqrtw
        P[:, a] .= res.v           
        sv[a] = res.sv
        niter[a] = res.niter
    end    
    Pca(T, P, sv, xmeans, xscales, weights, niter) 
end

