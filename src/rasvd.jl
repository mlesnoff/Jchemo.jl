"""
    rasvd(X, Y, weights = ones(nro(X)); nlv,
        bscal = :none, tau = 1e-8, scal::Bool = false)
    rasvd!(X, Y, weights = ones(nro(X)); nlv,
        bscal = :none, tau = 1e-8, scal::Bool = false)
Redundancy analysis (RA) - PCA on instrumental variables (PCAIV)
* `X` : First block of data (explicative variables).
* `Y` : Second block of data (dependent variables).
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling (`:none`, `:frob`). 
    See functions `blockscal`.
* `tau` : Regularization parameter (∊ [0, 1]).
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).
 
See e.g. Bougeard et al. 2011a,b and Legendre & Legendre 2012. 
Let Y_hat be the fitted values of the regression of `Y` on `X`. 
The scores `Ty` are the PCA scores of Y_hat. The scores `Tx` are 
the fitted values of the regression of `Ty` on `X`.

A continuum regularization is available.  
After block centering and scaling, the covariances matrices are 
computed as follows: 
* Cx = (1 - `tau`) * X'DX + `tau` * Ix
where D is the observation (row) metric. 
Value `tau` = 0 can generate unstability when inverting the covariance matrices. 
A better alternative is generally to use an epsilon value (e.g. `tau` = 1e-8) 
to get similar results as with pseudo-inverses.    

## References
Bougeard, S., Qannari, E.M., Lupo, C., Chauvin, C., 2011. Multiblock redundancy 
analysis from a user's perspective. Application in veterinary epidemiology. 
Electronic Journal of Applied Statistical Analysis 4, 203-214. 
https://doi.org/10.1285/i20705948v4n2p203

Bougeard, S., Qannari, E.M., Rose, N., 2011. Multiblock redundancy analysis: 
interpretation tools and application in epidemiology. Journal of Chemometrics 25, 
467-475. https://doi.org/10.1002/cem.1392

Legendre, P., Legendre, L., 2012. Numerical Ecology. Elsevier, 
Amsterdam, The Netherlands.

Tenenhaus, A., Guillemot, V. 2017. RGCCA: Regularized and Sparse Generalized Canonical 
Correlation Analysis for Multiblock Data Multiblock data analysis.
https://cran.r-project.org/web/packages/RGCCA/index.html 

## Examples
```julia
using JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/linnerud.jld2") 
@load db dat
pnames(dat)
X = dat.X 
Y = dat.Y

tau = 1e-8
fm = rasvd(X, Y; nlv = 3, tau = tau)
pnames(fm)

fm.Tx
transform(fm, X, Y).Tx
scale(fm.Tx, colnorm(fm.Tx))

res = summary(fm, X, Y)
pnames(res)
```
"""
function rasvd(X, Y, weights = ones(nro(X)); nlv,
        bscal = :none, tau = 1e-8, scal::Bool = false)
    rasvd!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv,
        bscal = bscal, tau = tau, scal = scal)
end

function rasvd!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        bscal = :none, tau = 1e-8, scal = scal)
    @assert 0 <= par.tau <=1 "tau must be in [0, 1]"
    Q = eltype(X)
    p = nco(X)
    q = nco(Y)
    nlv = min(nlv, p, q)
    weights = mweight(weights)
    sqrtw = sqrt.(weights.w)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        cscale!(X, xmeans, xscales)
        cscale!(Y, ymeans, yscales)
    else
        center!(X, xmeans)
        center!(Y, ymeans)
    end
    bscal == :none ? bscales = ones(2) : nothing
    if bscal == :frob
        normx = frob(X, weights)
        normy = frob(Y, weights)
        X ./= normx
        Y ./= normy
        bscales = [normx; normy]
    end
    # Row metric
    X .= sqrtw .* X
    Y .= sqrtw .* Y
    # End
    if tau == 0       
        invCx = inv(X' * X)
    else
        Ix = Diagonal(ones(Q, p)) 
        if tau == 1   
            invCx = Ix
        else
            invCx = inv((1 - tau) * X' * X + tau * Ix)
        end
    end
    Bx = invCx * X' * Y 
    Yfit = X * Bx
    res = LinearAlgebra.svd(Yfit)
    Wy = res.V[:, 1:nlv]    # = C
    lambda = res.S[1:nlv].^2
    Ty = Y * Wy
    Tx = Yfit * Wy    
    # Same as:
    # Projx = X * invCx * X'
    # Yfit = Projx * Y
    # PCA(Yfit) ==> Wy, Ty
    # Tx = Projx * Ty
    # End
    Tx .= (1 ./ sqrtw) .* Tx
    Ty .= (1 ./ sqrtw) .* Ty   
    Rasvd(Tx, Ty, Bx, Wy, lambda, 
        bscales, xmeans, xscales, ymeans, yscales, weights)
end

""" 
    transform(object::Rasvd, X, Y; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and (X, Y)-data.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute. If nothing, it is the maximum number
    from the fitted model.
""" 
function transform(object::Rasvd, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    X = cscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = cscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    Yfit = X * object.Bx
    Wy = vcol(object.Wy, 1:nlv)
    Tx = Yfit * Wy
    Ty = Y * Wy
    (Tx = Tx, Ty)
end

## Same as ::Cca
## But explvary has to be computed (To Do)
"""
    summary(object::Rasvd, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::Rasvd, X,
        Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = cscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = cscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    D = Diagonal(object.weights.w)
    # X
    T = object.Tx
    sstot = frob(X, object.weights)^2
    sst = diag(T' * D * X * X' * D * T) ./ diag(T' * D * T)
    pvar =  sst / sstot
    cumpvar = cumsum(pvar)
    xvar = sst / n
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, 
        cumpvar = cumpvar)
    # Y
    T .= object.Ty
    frob(Y, object.weights)^2
    sst = diag(T' * D * Y * Y' * D * T) ./ diag(T' * D * T)
    pvar =  sst / sstot
    cumpvar = cumsum(pvar)
    explvary = nothing # TO DO
    #explvary = DataFrame(nlv = 1:nlv, var = sst, pvar = pvar, cumpvar = cumpvar)
    # Correlation between X- and Y-block scores
    z = diag(corm(object.Tx, object.Ty, object.weights))
    cort2t = DataFrame(lv = 1:nlv, cor = z)
    # Redundancies (Average correlations) Rd(X, tx) and Rd(Y, ty)
    z = rd(X, object.Tx, object.weights)
    rdx = DataFrame(lv = 1:nlv, rd = vec(z))
    z = rd(Y, object.Ty, object.weights)
    rdy = DataFrame(lv = 1:nlv, rd = vec(z))
    # Correlation between block variables and their block scores
    z = corm(X, object.Tx, object.weights)
    corx2t = DataFrame(z, string.("lv", 1:nlv))
    z = corm(Y, object.Ty, object.weights)
    cory2t = DataFrame(z, string.("lv", 1:nlv))
    # End 
    (explvarx = explvarx, explvary, cort2t, rdx, rdy, 
        corx2t, cory2t)
end