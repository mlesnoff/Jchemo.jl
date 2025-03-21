"""
    pcanipals(; kwargs...)
    pcanipals(X; kwargs...)
    pcanipals(X, weights::Weight; kwargs...)
    pcanipals!(X::Matrix, weights::Weight; kwargs...)
PCA by NIPALS algorithm.
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. of principal components (PCs).
* `gs` : Boolean. If `true` (default), a Gram-Schmidt 
    orthogonalization of the scores and loadings is done
    before each X-deflation. 
* `tol` : Tolerance value for stopping 
    the iterations.
* `maxit` : Maximum nb. of iterations.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Let us note D the (n, n) diagonal matrix of weights
(`weights.w`) and X the centered matrix in metric D.
The function minimizes ||X - T * V'||^2  in metric D 
by NIPALS. 

See function `pcasvd` for examples.

## References
Andrecut, M., 2009. Parallel GPU Implementation of Iterative 
PCA Algorithms. Journal of Computational Biology 16, 1593-1599. 
https://doi.org/10.1089/cmb.2008.0221

K.R. Gabriel, S. Zamir, Lower rank approximation of matrices 
by least squares with any choice of weights, Technometrics 21 (1979)
489–498.

Gabriel, R. K., 2002. Le biplot - Outil d'exploration de données 
multidimensionnelles. Journal de la Société Française de la Statistique, 
143, 5-55.

Lingen, F.J., 2000. Efficient Gram-Schmidt orthonormalisation 
on parallel computers. Communications in Numerical Methods in 
Engineering 16, 57-66. 
https://doi.org/10.1002/(SICI)1099-0887(200001)16:1<57::AID-CNM320>3.0.CO;2-I

Tenenhaus, M., 1998. La régression PLS: théorie et pratique. 
Editions Technip, Paris, France.

Wright, K., 2018. Package nipals: Principal Components Analysis 
using NIPALS with Gram-Schmidt Orthogonalization. 
https://cran.r-project.org/
""" 
pcanipals(; kwargs...) = JchemoModel(pcanipals, nothing, kwargs)

function pcanipals(X; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    pcanipals(X, weights; kwargs...)
end

function pcanipals(X, weights::Weight; kwargs...)
    pcanipals!(copy(ensure_mat(X)), weights; kwargs...)
end

function pcanipals!(X::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParPcanipals, kwargs).par
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    xmeans = colmean(X, weights) 
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    sqrtw = sqrt.(weights.w)
    fweight!(X, sqrtw)
    t = similar(X, n)
    T = similar(X, n, nlv)
    V = similar(X, p, nlv)
    sv = similar(X, nlv)
    niter = list(Int, nlv)
    if par.gs
        UUt = zeros(Q, n, n)
        VVt = zeros(Q, p, p)
    end
    for a = 1:nlv
        if par.gs
            res = nipals(X, UUt, VVt; kwargs...)
        else
            res = nipals(X; kwargs...)
        end
        t .= res.u * res.sv
        T[:, a] .= t 
        V[:, a] .= res.v           
        sv[a] = res.sv
        niter[a] = res.niter
        X .-= t * res.v'
        if par.gs
            UUt .+= res.u * res.u' 
            VVt .+= res.v * res.v'
        end
    end
    fweight!(T, 1 ./ sqrtw)    
    ## Could recompute the scores by
    ## X0 = copy(X) ; ... ; T = (1 ./ sqrtw) .* X0 * V 
    Pca(T, V, sv, xmeans, xscales, weights, niter, par) 
end

