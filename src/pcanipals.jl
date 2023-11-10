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

Let us note D the (n, n) diagonal matrix of `weights` and X the centered 
matrix in metric D. The function minimizes ||X - T * P'||^2  in metric D 
by NIPALS. 

See `?pcasvd` for examples.

## References
Andrecut, M., 2009. Parallel GPU Implementation of Iterative PCA Algorithms. 
Journal of Computational Biology 16, 1593-1599. https://doi.org/10.1089/cmb.2008.0221

K.R. Gabriel, S. Zamir, Lower rank approximation of matrices by least squares with 
any choice of weights, Technometrics 21 (1979) 489–498

Gabriel, R. K., 2002. Le biplot - Outil d'exploration de données multidimensionnelles. 
Journal de la Société Française de la Statistique, 143, 5-55.

Lingen, F.J., 2000. Efficient Gram-Schmidt orthonormalisation on parallel computers. 
Communications in Numerical Methods in Engineering 16, 57-66. 
https://doi.org/10.1002/(SICI)1099-0887(200001)16:1<57::AID-CNM320>3.0.CO;2-I

Tenenhaus, M., 1998. La régression PLS: théorie et pratique. 
Editions Technip, Paris, France.

Wright, K., 2018. Package nipals: Principal Components Analysis using NIPALS 
with Gram-Schmidt Orthogonalization. https://cran.r-project.org/
""" 
function pcanipals(X; par = Par())
    weights = mweight(ones(eltype(X), nro(X)))
    pcanipals!(copy(ensure_mat(X)), weights; par)
end

function pcanipals(X, weights::Vector{Q}; 
        par = Par()) where {Q <: AbstractFloat}
    pcanipals!(copy(ensure_mat(X)), weights; par)
end

function pcanipals!(X::Matrix, weights::Vector{Q}; 
        par = Par()) where {Q <: AbstractFloat}
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    xmeans = colmean(X, weights) 
    xscales = ones(eltype(X), p)
    if par.scal 
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
    niter = list(nlv, Int)
    if par.gs
        UUt = zeros(n, n)
        VVt = zeros(p, p)
    end
    for a = 1:nlv
        if par.gs == false
            res = nipals(X; tol = par.tol, maxit = par.maxit)
        else
            res = nipals(X, UUt, VVt; 
                tol = par.tol, maxit = par.maxit)
        end
        t .= res.u * res.sv
        T[:, a] .= t ./ sqrtw
        P[:, a] .= res.v           
        sv[a] = res.sv
        niter[a] = res.niter
        X .-= t * res.v'
        if par.gs
            UUt .+= res.u * res.u' 
            VVt .+= res.v * res.v'
        end
    end    
    ## Could recompute the scores by
    ## X0 = copy(X) ; ... ; T = (1 ./ sqrtw) .* X0 * P 
    Pca(T, P, sv, xmeans, xscales, weights, niter) 
end

