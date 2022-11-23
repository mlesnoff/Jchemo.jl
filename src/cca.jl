struct Cca
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
    cca(X, Y, weights = ones(nro(X)); nlv, 
        bscal = "none", tau = 0, scal = false)
    cca!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        bscal = "none", tau = 0, scal = false)
Regularized canonical correlation Analysis (RCCA).
* `X` : First block (matrix) of data.
* `Y` : Second block (matrix) of data.
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling (`"none"`, `"frob"`). 
    See functions `blockscal`.
* `tau` : Regularization parameter (∊ [0, 1]).
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

The CCA approach used in this function is presented in Weenink 2003. 
When the observation weights are uniform, the normed scores returned 
by the function are the same as those returned by functions `rcc` of 
the R packages `CCA` (González et al.) and `mixOmics` (Le Cao et al.) 
whith the parameters lambda1 and lambda2 set to:
* lambda = `tau` / (1 - `tau`) * n / (n - 1) 

The regularization uses the continuum formulation presented by 
Qannari & Hanafi 2005 and Mangamana et al. 2019. 
After block centering and scaling, the block scores returned by 
the present algorithm (Tx and Ty) are proportionnal to the eigenvectors of 
Projx * Projy and Projy * Projx, respectively, defined as follows: 
* Cx = (1 - `tau`) * X'DX + `tau` * Ix
* Cy = (1 - `tau`) * Y'DY + `tau` * Iy
* Cxy = X'DY
* Projx = sqrt(D) * X * invCx * X' * sqrt(D)
* Projy = sqrt(D) * Y * invCx * Y' * sqrt(D)
where D is the observation (row) metric. 
The final scores are returned in the original scale, 
by multiplying by D^(-1/2).

## References
González, I., Déjean, S., Martin, P.G.P., Baccini, A., 2008. CCA: 
An R Package to Extend Canonical Correlation Analysis. Journal of Statistical 
Software 23, 1–14. https://doi.org/10.18637/jss.v023.i12

Hotelling, H. (1936): “Relations between two sets of variates”, Biometrika 28: pp. 321–377.

Qannari, E.M., Hanafi, M., 2005. A simple continuum regression approach. 
Journal of Chemometrics 19, 387–392. https://doi.org/10.1002/cem.942

Le Cao, K.-A., Rohart, F., Gonzalez, I., Dejean, S., Abadi, A.J., Gautier, B., Bartolo, F., 
Monget, P., Coquery, J., Yao, F., Liquet, B., 2022. mixOmics: Omics Data Integration Project. 
https://doi.org/10.18129/B9.bioc.mixOmics

Tchandao Mangamana, E., Cariou, V., Vigneau, E., Glèlè Kakaï, R.L., Qannari, E.M., 2019. 
Unsupervised multiblock data analysis: A unified approach and extensions. 
Chemometrics and Intelligent Laboratory Systems 194, 103856. 
https://doi.org/10.1016/j.chemolab.2019.103856

Weenink, D. 2003. Canonical Correlation Analysis, Institute of Phonetic Sciences, 
Univ. of Amsterdam, Proceedings 25, 81-99.

## Examples
```julia
using JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "linnerud.jld2") 
@load db dat
pnames(dat)
X = dat.X 
Y = dat.Y

tau = 0
fm = cca(X, Y; nlv = 3, tau = tau)
pnames(fm)

fm.Tx
transform(fm, X, Y).Tx
scale(fm.Tx, colnorm(fm.Tx))

res = summary(fm, X, Y)
pnames(res)
```
"""
function cca(X, Y, weights = ones(nro(X)); nlv, 
        bscal = "none", tau = 0, scal = false)
    cca!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv, 
        bscal = bscal, tau = tau, scal = scal)
end

function cca!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        bscal = "none", tau = 0, scal = false)
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
    if tau == 0
        Cx = Symmetric(X' * X)
        Cy = Symmetric(Y' * Y)
    else
        Ix = Diagonal(ones(p)) 
        Iy = Diagonal(ones(q)) 
        if tau == 1
            Cx = Ix
            Cy = Iy
        else
            Cx = Symmetric((1 - tau) * X' * X + tau * Ix)
            Cy = Symmetric((1 - tau) * Y' * Y + tau * Iy)
        end
    end
    Cxy = X' * Y    
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
    Cca(Tx, Ty, Wx, Wy, d, 
        bscales, xmeans, xscales, ymeans, yscales, weights)
end

""" 
    transform(object::Cca, X, Y; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and (X, Y)-data.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute. If nothing, it is the maximum number
    from the fitted model.
""" 
function transform(object::Cca, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    Tx = cscale(X, object.xmeans, object.xscales) * vcol(object.Wx, 1:nlv)
    Ty = cscale(Y, object.ymeans, object.yscales) * vcol(object.Wy, 1:nlv)
    (Tx = Tx, Ty)
end

"""
    summary(object::Cca, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::Cca, X::Union{Vector, Matrix, DataFrame},
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

