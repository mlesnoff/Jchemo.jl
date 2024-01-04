"""
    cca(; kwargs...)
    cca(X, Y; kwargs...)
    cca(X, Y, weights::Weight; 
        kwargs...)
    cca!(X::Matrix, Y::Matrix, weights::Weight; 
        kwargs...)
Canonical correlation Analysis (CCA, RCCA).
* `X` : First block of data.
* `Y` : Second block of data.
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
Keyword arguments:
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling. Possible values are:
    `:none`, `:frob`. See functions `fblockscal`.
* `tau` : Regularization parameter (∊ [0, 1]).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

This function implements a CCA algorithm using SVD decompositions and 
presented in Weenink 2003 section 2. 

A continuum regularization is available (parameter `tau`). 
After block centering and scaling, the function returns 
block scores (Tx and Ty) that are proportionnal to the 
eigenvectors of Projx * Projy and Projy * Projx, respectively, 
defined as follows: 
* Cx = (1 - `tau`) * X'DX + `tau` * Ix
* Cy = (1 - `tau`) * Y'DY + `tau` * Iy
* Cxy = X'DY 
* Projx = sqrt(D) * X * invCx * X' * sqrt(D)
* Projy = sqrt(D) * Y * invCx * Y' * sqrt(D)
where D is the observation (row) metric. 
Value `tau` = 0 can generate unstability when inverting 
the covariance matrices. Often, a better alternative is 
to use an epsilon value (e.g. `tau` = 1e-8) to get similar 
results as with pseudo-inverses.  

The normed scores returned by the function are expected 
(using uniform `weights`) to be the same as those 
returned by functions `rcc` of the R packages `CCA` (González et al.) 
and `mixOmics` (Lê Cao et al.) whith their parameters lambda1 
and lambda2 set to:
* lambda1 = lambda2 = `tau` / (1 - `tau`) * n / (n - 1) 

## References
González, I., Déjean, S., Martin, P.G.P., Baccini, A., 2008. 
CCA: An R Package to Extend Canonical Correlation Analysis. 
Journal of Statistical Software 23, 1-14. 
https://doi.org/10.18637/jss.v023.i12

Hotelling, H. (1936): “Relations between two sets of variates”, 
Biometrika 28: pp. 321–377.

Lê Cao, K.-A., Rohart, F., Gonzalez, I., Dejean, S., Abadi, A.J., 
Gautier, B., Bartolo, F., Monget, P., Coquery, J., Yao, F., 
Liquet, B., 2022. mixOmics: Omics Data Integration Project. 
https://doi.org/10.18129/B9.bioc.mixOmics

Weenink, D. 2003. Canonical Correlation Analysis, Institute of 
Phonetic Sciences, Univ. of Amsterdam, Proceedings 25, 81-99.

## Examples
```julia
using JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "linnerud.jld2") 
@load db dat
pnames(dat)
X = dat.X
Y = dat.Y
n, p = size(X)
q = nco(Y)

nlv = 3
bscal = :frob ; tau = 1e-8
mod = cca(; nlv, bscal, 
    tau)
fit!(mod, X, Y)
pnames(mod)
pnames(mod.fm)

@head mod.fm.Tx
@head transfbl(mod, X, Y).Tx

@head mod.fm.Ty
@head transfbl(mod, X, Y).Ty

res = summary(mod, X, Y) ;
pnames(res)
res.cort2t 
res.rdx
res.rdy
res.corx2t 
res.cory2t 
```
"""
function cca(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    n = nro(X)
    weights = mweight(ones(Q, n))
    cca(X, Y, weights; kwargs...)
end

function cca(X, Y, weights::Weight; 
        kwargs...)
    cca!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; kwargs...)
end

function cca!(X::Matrix, Y::Matrix, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs) 
    @assert in([:none, :frob])(par.bscal) "Wrong value for argument 'bscal'."
    @assert 0 <= par.tau <= 1 "tau must be in [0, 1]"
    Q = eltype(X)
    p = nco(X)
    q = nco(Y)
    nlv = min(par.nlv, p, q)
    tau = convert(Q, par.tau) 
    sqrtw = sqrt.(weights.w)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        fcscale!(X, xmeans, xscales)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
    end
    par.bscal == :none ? bscales = ones(Q, 2) : nothing
    if par.bscal == :frob
        normx = frob(X, weights)
        normy = frob(Y, weights)
        X ./= normx
        Y ./= normy
        bscales = [normx ; normy]
    end
    # Row metric
    X .= sqrtw .* X
    Y .= sqrtw .* Y 
    # End
    if tau == 0
        Cx = Symmetric(X' * X)
        Cy = Symmetric(Y' * Y)
    else
        Ix = Diagonal(ones(Q, p)) 
        Iy = Diagonal(ones(Q, q)) 
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
    Cca(Tx, Ty, Wx, Wy, d, bscales, xmeans, xscales, 
        ymeans, yscales, weights, kwargs, par)
end

""" 
    transfbl(object::Cca, X, Y; 
        nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transfbl(object::Cca, X, Y; 
        nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    X = fcscale(X, object.xmeans, 
        object.xscales) / object.bscales[1]
    Y = fcscale(Y, object.ymeans, 
        object.yscales) / object.bscales[2]
    Tx = X * vcol(object.Wx, 1:nlv)
    Ty = Y * vcol(object.Wy, 1:nlv)
    (Tx = Tx, Ty)
end

"""
    summary(object::Cca, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::Cca, X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = fcscale(X, object.xmeans, 
        object.xscales) / object.bscales[1]
    Y = fcscale(Y, object.ymeans, 
        object.yscales) / object.bscales[2]
    ## To do: explvarx, explvary 
    #D = Diagonal(object.weights.w)   
    ### X
    #T = object.Tx
    #sstot = frob(X, object.weights)^2
    #tt = diag(T' * D * X * X' * D * T) ./ diag(T' * D * T)
    ##tt = colsum(D * T .* T)  # = [1]
    #pvar =  tt / sstot
    #cumpvar = cumsum(pvar)
    #xvar = tt / n    
    #explvarx = DataFrame(nlv = 1:nlv, var = xvar, 
    #    pvar = pvar, cumpvar = cumpvar)
    explvarx = nothing 
    ### Y
    #T = object.Ty
    #sstot = frob(Y, object.weights)^2
    #tt = diag(T' * D * Y * Y' * D * T) ./ diag(T' * D * T)
    ##tt = colsum(D * T .* T)  # = [1]
    #pvar =  tt / sstot
    #cumpvar = cumsum(pvar)
    #xvar = tt / n    
    #explvary = DataFrame(nlv = 1:nlv, var = xvar, 
    #    pvar = pvar, cumpvar = cumpvar)
    explvary = nothing
    ## Correlation between X- and 
    ## Y-block scores
    z = diag(corm(object.Tx, object.Ty, 
        object.weights))
    cort2t = DataFrame(lv = 1:nlv, cor = z)
    ## Redundancies (Average correlations) 
    ## Rd(X, tx) and Rd(Y, ty)
    z = rd(X, object.Tx, object.weights)
    rdx = DataFrame(lv = 1:nlv, rd = vec(z))
    z = rd(Y, object.Ty, object.weights)
    rdy = DataFrame(lv = 1:nlv, rd = vec(z))
    ## Correlation between block variables 
    ## and their block scores
    z = corm(X, object.Tx, object.weights)
    corx2t = DataFrame(z, string.("lv", 1:nlv))
    z = corm(Y, object.Ty, object.weights)
    cory2t = DataFrame(z, string.("lv", 1:nlv))
    ## End
    (explvarx = explvarx, explvary, cort2t, 
        rdx, rdy, corx2t, cory2t)
end

