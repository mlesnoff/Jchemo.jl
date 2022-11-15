struct Rcca
    Tx::Matrix{Float64}
    Ty::Matrix{Float64}
    Wx::Matrix{Float64}
    Wy::Matrix{Float64}
    d::Vector{Float64}    
    bscales::Vector{Float64}    
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end


"""
    rcca(X, Y, weights = ones(nro(X)); nlv, 
        bscal = "none", alpha = 0, scal = false)
    rcca!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        bscal = "none", alpha = 0, scal = false)
Regularized canonical correlation Analysis (RCCA)
* `X` : First block (matrix) of data.
* `Y` : Second block (matrix) of data.
* `weights` : Weights of the observations (rows). Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscal` : Type of block scaling (`"none"`, `"frob"`). 
    See functions `blockscal`.
* `tol` : Tolerance value for convergence.
* `niter` : Maximum number of iterations.
* `scal` : Boolean. If `true`, each column of `Xbl` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

This version corresponds to the "SVD" algorithm of Hannafi & Qannari 2008 p.84.



The function returns several objects, in particular:
* `T` : The non normed global scores.
* `U` : The normed global scores.
* `W` : The global loadings.
* `Tb` : The block scores.
* `Wbl` : The block loadings.
* `lb` : The specific weights (saliences) "lambda".
* `mu` : The sum of the squared saliences.

Function `summary` returns: 
* `explvarx` : Proportion of the X total inertia (sum of the squared norms of the 
    blocks) explained by each global score.
* `explvarxx` : Proportion of the XX' total inertia (sum of the squared norms of the
    products X_k * X_k') explained by each global score 
    (= indicator "V" in Qannari et al. 2000, Hanafi et al. 2008).
* `sal2` : Proportion of the squared saliences (specific weights)
    of each block within each global score. 
* `contr_block` : Contribution of each block to the global scores 
    (= proportions of the saliences "lambda" within each score)
* `explX` : Proportion of the inertia of the blocks explained by each global score.
* `cort2x` : Correlation between the global scores and the original variables.  
* `cort2tb` : Correlation between the global scores and the block scores.
* `rv` : RV coefficient. 
* `lg` : Lg coefficient. 

## References
Cariou, V., Qannari, E.M., Rutledge, D.N., Vigneau, E., 2018. ComDim: From multiblock data 
analysis to path modeling. Food Quality and Preference, Sensometrics 2016: 
Sensometrics-by-the-Sea 67, 27–34. https://doi.org/10.1016/j.foodqual.2017.02.012

Cariou, V., Jouan-Rimbaud Bouveresse, D., Qannari, E.M., Rutledge, D.N., 2019. 
Chapter 7 - ComDim Methods for the Analysis of Multiblock Data in a Data Fusion 
Perspective, in: Cocchi, M. (Ed.), Data Handling in Science and Technology, 
Data Fusion Methodology and Applications. Elsevier, pp. 179–204. 
https://doi.org/10.1016/B978-0-444-63984-4.00007-7

Ghaziri, A.E., Cariou, V., Rutledge, D.N., Qannari, E.M., 2016. Analysis of multiblock 
datasets using ComDim: Overview and extension to the analysis of (K + 1) datasets. 
Journal of Chemometrics 30, 420–429. https://doi.org/10.1002/cem.2810

Hanafi, M., 2008. Nouvelles propriétés de l’analyse en composantes communes et 
poids spécifiques. Journal de la société française de statistique 149, 75–97.

Qannari, E.M., Wakeling, I., Courcoux, P., MacFie, H.J.H., 2000. Defining the underlying 
sensory dimensions. Food Quality and Preference 11, 151–154. 
https://doi.org/10.1016/S0950-3293(99)00069-5

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "ham.jld2") 
@load db dat
pnames(dat) 

X = dat.X
group = dat.group
listbl = [1:11, 12:19, 20:25]
Xbl = mblock(X, listbl)
# "New" = first two rows of Xbl 
Xbl_new = mblock(X[1:2, :], listbl)

bscal = "none"
#bscal = "frob"
fm = comdim(Xbl; nlv = 4, bscal = bscal) ;
fm.U
fm.T
Jchemo.transform(fm, Xbl)
Jchemo.transform(fm, Xbl_new) 

res = Jchemo.summary(fm, Xbl) ;
fm.lb
rowsum(fm.lb)
fm.mu
res.explvarx
res.explvarxx
res.explX # = fm.lb if bscal = "frob"
rowsum(Matrix(res.explX))
res.contr_block
res.sal2
colsum(Matrix(res.sal2))
res.cort2x 
res.cort2tb
res.rv
```
"""
function rcca(X, Y, weights = ones(nro(X)); nlv, 
        bscal = "none", alpha = 0, scal = false)
    rcca!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv, 
        bscal = bscal, alpha = alpha, scal = scal)
end

function rcca!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        bscal = "none", alpha = 0, scal = false)
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, nlv)
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    xscales = ones(p)
    yscales = ones(q)
    if scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        cscale!(X, xmeans, xscales)
        cscale!(Y, ymeans, yscales)
    else
        center!(X, xmeans)
        center!(Y, ymeans)
    end
    bscal == "none" ? bscales = ones(2) : nothing
    if bscal == "frob"
        normx = fnorm(X, weights)
        normy = fnorm(Y, weights)
        X ./= normx
        Y ./= normy
        bscales = [normx; normy]
    end
    # Row metric
    sqrtw = sqrt.(weights)
    X .= sqrtw .* X
    Y .= sqrtw .* Y 
    # End
    if alpha == 0
        Cx = Symmetric(X' * X)
        Cy = Symmetric(Y' * Y)
    else
        Ix = Diagonal(ones(p))
        Iy = Diagonal(ones(q))
        if alpha == 1
            Cx = Ix
            Cy = Iy
        else
            Cx = Symmetric((1 - alpha) * X' * X + alpha * Ix)
            Cy = Symmetric((1 - alpha) * Y' * Y + alpha * Iy)
        end
    end
    Cxy = X'Y        
    Ux = cholesky(Hermitian(Cx)).U
    Uy = cholesky(Hermitian(Cy)).U
    invUx = inv(Ux)
    invUy = inv(Uy)
    A = invUx' * Cxy * invUy
    U, d, V = svd(A)
    Wx = invUx * U[:, 1:nlv]
    Wy = invUy * V[:, 1:nlv]
    d = d[1:nlv]
    Tx = (1 ./ sqrtw) .* X * Wx
    Ty = (1 ./ sqrtw) .* Y * Wy
    Rcca(Tx, Ty, Wx, Wy, d, 
        bscales, xmeans, xscales, ymeans, yscales, weights)
end

""" 
    transform(object::Rcca, X, Y; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and (X, Y)-data.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute. If nothing, it is the maximum number
    from the fitted model.
""" 
function transform(object::Rcca, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    Tx = cscale(X, object.xmeans, object.xscales) * vcol(object.Wx, 1:nlv)
    Ty = cscale(Y, object.ymeans, object.yscales) * vcol(object.Wy, 1:nlv)
    (Tx = Tx, Ty)
end

"""
    summary(object::Rcca, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::Rcca, X::Union{Vector, Matrix, DataFrame},
        Y::Union{Vector, Matrix, DataFrame})
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    nlv = nco(object.Tx)
    X = cscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = cscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    D = Diagonal(object.weights)
    ## Explained variances
    T = object.Tx
    xvar = diag(T' * D * X * X' * D * T) ./ diag(T' * D * T)
    pvar =  xvar / fnorm(X, object.weights)^2
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    T = object.Ty
    xvar = diag(T' * D * Y * Y' * D * T) ./ diag(T' * D * T)
    pvar =  xvar / fnorm(Y, object.weights)^2
    cumpvar = cumsum(pvar)
    explvary = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## Correlation between block scores
    z = diag(corm(object.Tx, object.Ty, object.weights))
    cort2t = DataFrame(lv = 1:nlv, cor = z)
    ## Redundancies (Average correlations)
    z = rd(X, object.Tx, object.weights)
    rdx = DataFrame(lv = 1:nlv, rd = vec(z))
    z = rd(Y, object.Ty, object.weights)
    rdy = DataFrame(lv = 1:nlv, rd = vec(z))
    ## Correlation between block variables and block scores
    z = corm(X, object.Tx, object.weights)
    corx2t = DataFrame(z, string.("lv", 1:nlv))
    z = corm(Y, object.Ty, object.weights)
    cory2t = DataFrame(z, string.("lv", 1:nlv))
    ## End
    (explvarx = explvarx, explvary, cort2t, rdx, rdy, 
        corx2t, cory2t)
end

