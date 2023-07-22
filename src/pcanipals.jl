"""
    pcanipals(X, weights = ones(nro(X)); nlv, scal::Bool = false)
    pcanipals!(X::Matrix, weights = ones(nro(X)); nlv, scal::Bool = false)
PCA by NIPALS algorithm.
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. principal components (PCs).
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

Let us note D the (n, n) diagonal matrix of `weights`
and X the centered matrix in metric D. 
The function minimizes ||X - T * P'||^2  in metric D. 

See `?pcasvd` for examples.
""" 

function pcanipals(X, weights = ones(nro(X)); nlv, 
        gs = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
    pcanipals!(copy(ensure_mat(X)), weights; nlv = nlv, 
        gs = gs, tol = tol, maxit = maxit, scal = scal)
end

function pcanipals!(X::Matrix, weights = ones(nro(X)); nlv, 
        gs = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
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
        T[:, a] .= t ./ sqrtw
        X .-= t * res.v'
        niter[a] = res.niter
    end    
    Pca(T, P, sv, xmeans, xscales, weights, niter) 
end

