function pcanipalsmiss(X, weights = ones(nro(X)); nlv, 
        gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
    pcanipalsmiss!(copy(ensure_mat(X)), weights; nlv = nlv, 
        gs = gs, tol = tol, maxit = maxit, scal = scal)
end

function pcanipalsmiss!(X::Matrix, weights = ones(nro(X)); nlv, 
        gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
    n, p = size(X)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
    xmeans = colmeanskip(X, weights) 
    #xmeans = colmeanskip(X)
    xscales = ones(p)
    if scal 
        xscales .= colstdskip(X, weights)
        #xscales .= colstdskip(X)
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
    if gs
        UUt = zeros(n, n)
        VVt = zeros(p, p)
    end
    for a = 1:nlv
        if gs == false
            res = nipalsmiss(X; tol = tol, maxit = maxit)
        else
            res = nipalsmiss(X, UUt, VVt; 
                tol = tol, maxit = maxit)
        end
        t .= res.u * res.sv
        T[:, a] .= t ./ sqrtw
        #T[:, a] .= t
        P[:, a] .= res.v           
        sv[a] = res.sv
        niter[a] = res.niter
        X .-= t * res.v'
        if gs
            UUt .+= res.u * res.u' 
            VVt .+= res.v * res.v'
        end
    end    
    Pca(T, P, sv, xmeans, xscales, weights, niter) 
end

