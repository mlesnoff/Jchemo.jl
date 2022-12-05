struct RaSvd
    Tx::Matrix{Float64}
    Ty::Matrix{Float64}
    Bx::Matrix{Float64}
    Wy::Matrix{Float64}
    lambda::Vector{Float64}    
    bscales::Vector{Float64}    
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end

"""
    rasvd(X, Y, weights = ones(nro(X)); nlv,
        bscal = "none", tau = 1e-10, scal = false)
    rasvd!(X, Y, weights = ones(nro(X)); nlv,
        bscal = "none", tau = 1e-10, scal = false)
Redundancy analysis (RRA) - SVD aglorithm (PCA-IV).
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

The regularization uses the continuum formulation presented by 
Qannari & Hanafi 2005 and Mangamana et al. 2019. After block centering and scaling, 
the covariances matrices are computed as follows: 
* Cx = (1 - `tau`) * X'DX + `tau` * Ix
* Cy = (1 - `tau`) * Y'DY + `tau` * Iy
where D is the observation (row) metric. 
The final scores are returned in the original scale, 
by multiplying by D^(-1/2).

## References
Bougeard, S., Qannari, E.M., Lupo, C., Chauvin, C., 2011a. Multiblock redundancy 
analysis from a user’s perspective. Application in veterinary epidemiology. 
Electronic Journal of Applied Statistical Analysis 4, 203-214–214. 
https://doi.org/10.1285/i20705948v4n2p203

Bougeard, S., Qannari, E.M., Rose, N., 2011b. Multiblock redundancy analysis: 
interpretation tools and application in epidemiology. Journal of Chemometrics 25, 
467–475. https://doi.org/10.1002/cem.1392

## Examples
```julia
using JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "linnerud.jld2") 
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
        bscal = "none", tau = 1e-10, scal = false)
    rasvd!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv,
        bscal = bscal, tau = tau, scal = scal)
end

function rasvd!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        bscal = "none", tau = 1e-10, scal = scal)
    n, p = size(X)
    q = nco(Y)
    nlv = min(nlv, n, p)
    weights = mweight(weights)
    sqrtw = sqrt.(weights)
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
        Ix = Diagonal(ones(p)) 
        if tau == 1   
            invCx = Ix
        else
            invCx = inv((1 - tau) * X' * X + tau * Ix)
        end
    end
    Bx = invCx * X' * Y 
    Yfit = X * Bx
    #Projx = X * invCx * X'
    #Yfit = Projx * Y
    res = LinearAlgebra.svd(Yfit)
    Wy = res.V[:, 1:nlv]    # = C
    lambda = res.S[1:nlv].^2
    Ty = Y * Wy
    Tx = Yfit * Wy    # = Projx * Ty
    #Tx = Projx * Ty
    Tx .= (1 ./ sqrtw) .* Tx
    Ty .= (1 ./ sqrtw) .* Ty   
    RaSvd(Tx, Ty, Bx, Wy, lambda, 
        bscales, xmeans, xscales, ymeans, yscales, weights)
end

""" 
    transform(object::RaSvd, X, Y; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and (X, Y)-data.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute. If nothing, it is the maximum number
    from the fitted model.
""" 
function transform(object::RaSvd, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    X = cscale(X, object.xmeans, object.xscales)
    Y = cscale(Y, object.ymeans, object.yscales)
    Yfit = X * object.Bx
    Wy = vcol(object.Wy, 1:nlv)
    Tx = Yfit * Wy
    Ty = Y * Wy
    (Tx = Tx, Ty)
end

## Same as ::Cca
## But explvary has to be computed (To Do)
"""
    summary(object::RaSvd, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::RaSvd, X::Union{Vector, Matrix, DataFrame},
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
    pvar =  xvar / frob(X, object.weights)^2
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    T .= object.Ty
    xvar = diag(T' * D * Y * Y' * D * T) ./ diag(T' * D * T)
    pvar =  xvar / frob(Y, object.weights)^2
    cumpvar = cumsum(pvar)
    explvary = nothing # TO DO
    #explvary = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
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