"""
    pcanipals(X, weights = ones(nro(X)); nlv, 
        gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
    pcanipals!(X::Matrix, weights = ones(nro(X)); nlv, 
        gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
PCA by NIPALS algorithm.
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. principal components (PCs).
* `gs` : Boolean. If `true` (default), a Gram-Schmidt orthogonalization 
    of the scores and loadings is done. 
* `tol` : Tolerance value for stopping the iterations.
* `maxit` : Maximum nb. iterations.
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

Let us note D the (n, n) diagonal matrix of `weights`
and X the centered matrix in metric D. 
The function minimizes ||X - T * P'||^2  in metric D by
NIPALS (see function `nipals`). 

See `?pcasvd` for examples.
""" 

function pcanipals(X, weights = ones(nro(X)); nlv, 
        gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
    pcanipals!(copy(ensure_mat(X)), weights; nlv = nlv, 
        gs = gs, tol = tol, maxit = maxit, scal = scal)
end

function pcanipals!(X::Matrix, weights = ones(nro(X)); nlv, 
        gs::Bool = true, tol = sqrt(eps(1.)), maxit = 200, 
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
    u = copy(t)
    u0 = copy(t)
    v = similar(X, p)
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    sv = similar(X, nlv)
    niter = list(nlv, Int64)
    if gs
        UUt = zeros(n, n)
        VVt = zeros(p, p)
    end
    for a = 1:nlv
        u .= X[:, argmax(colnorm(X))]
        cont = true
        iter = 1
        while cont
            u0 .= copy(u)      
            mul!(v, X', u)
            v ./= norm(v)
            if gs & (a > 1)
                v .= v .- VVt * v
            end
            mul!(u, X, v)
            if gs & (a > 1)
                u .= u .- UUt * u 
            end
            dif = sum((u .- u0).^2)
            iter = iter + 1
            if (dif < tol) || (iter > maxit)
                cont = false
            end
        end
        s = norm(u)
        u ./= s
        t .= u * s
        T[:, a] .= t ./ sqrtw
        P[:, a] .= v           
        sv[a] = s
        niter[a] = iter - 1
        X .-= t * v'
        if gs
            UUt .+= u * u' 
            VVt .+= v * v'
        end
    end    
    Pca(T, P, sv, xmeans, xscales, weights, niter) 
end

