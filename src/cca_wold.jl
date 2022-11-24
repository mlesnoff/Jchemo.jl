struct CcaWold
    Tx::Matrix{Float64}
    Ty::Matrix{Float64}
    Px::Matrix{Float64}
    Py::Matrix{Float64}
    Rx::Matrix{Float64}
    Ry::Matrix{Float64}    
    Wx::Matrix{Float64}
    Wy::Matrix{Float64}
    TTx::Vector{Float64}
    TTy::Vector{Float64}  
    bscales::Vector{Float64}    
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
    niter::Vector{Float64}
end

"""
    cca_wold(X, Y, weights = ones(nro(X)); nlv,
        bscal = "none", tau = 1e-10, 
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
    cca_wold!(X, Y, weights = ones(nro(X)); nlv,
        bscal = "none", tau = 1e-10, 
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
Regularized canonical correlation Analysis (RCCA) - Wold Nipals algorithm.
* `X` : First block (matrix) of data.
* `Y` : Second block (matrix) of data.
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling (`"none"`, `"frob"`). 
    See functions `blockscal`.
* `tau` : Regularization parameter (∊ [0, 1]).
* `tol` : Tolerance for the Nipals algorithm.
* `maxit` : Maximum number of iterations for the Nipals algorithm.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

The CCA approach used in this function is presented by Tenenhaus 1998 p.204
(Wold et al. 1984). 

The regularization uses the continuum formulation presented by 
Qannari & Hanafi 2005 and Mangamana et al. 2019. After block centering and scaling, 
the covariances matrices are computed as follows: 
* Cx = (1 - `tau`) * X'DX + `tau` * Ix
* Cy = (1 - `tau`) * Y'DY + `tau` * Iy
where D is the observation (row) metric. 
The final scores are returned in the original scale, 
by multiplying by D^(-1/2).

When the observation weights are uniform, the normed scores returned 
by the function are expected to be the same as those returned by functions `rgcca` of 
the R package `RGCCA` (Tenenhaus & Guillemot 2017, Tenenhaus et al. 2017). 
Nevertheless, the present Wold Nipals algorithm can show some unstability 
for values `tau` = 0. For more stability, it is recommended to replace 0 by 
an epsilon value (e.g. 1e-10) to get the same as with a pseudo-inverse.  

## References
Tenenhaus, A., Guillemot, V. 2017. RGCCA: Regularized and Sparse Generalized Canonical 
Correlation Analysis for Multiblock Data Multiblock data analysis.
https://cran.r-project.org/web/packages/RGCCA/index.html 

Tenenhaus, M., 1998. La régression PLS: théorie et pratique. Editions Technip, Paris.

Tenenhaus, M., Tenenhaus, A., Groenen, P.J.F., 2017. 
Regularized Generalized Canonical Correlation Analysis: A Framework for Sequential 
Multiblock Component Methods. Psychometrika 82, 737–777. 
https://doi.org/10.1007/s11336-017-9573-x

Wold, S., Ruhe, A., Wold, H., Dunn, III, W.J., 1984. The Collinearity Problem in Linear 
Regression. The Partial Least Squares (PLS) Approach to Generalized Inverses. 
SIAM Journal on Scientific and Statistical Computing 5, 735–743. 
https://doi.org/10.1137/0905052

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
fm = cca_wold(X, Y; nlv = 3, tau = tau)
pnames(fm)

fm.Tx
transform(fm, X, Y).Tx
scale(fm.Tx, colnorm(fm.Tx))

res = summary(fm, X, Y)
pnames(res)
```
"""
function cca_wold(X, Y, weights = ones(nro(X)); nlv,
        bscal = "none", tau = 1e-10, 
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
    cca_wold!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv,
        bscal = bscal, tau = tau, 
        tol = tol, maxit = maxit, scal = scal)
end

function cca_wold!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        bscal = "none", tau = 1e-10, 
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
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
        normx = fnorm(X, weights)
        normy = fnorm(Y, weights)
        X ./= normx
        Y ./= normy
        bscales = [normx; normy]
    end
    # Row metric
    X .= sqrtw .* X
    Y .= sqrtw .* Y
    # Pre-allocation
    Tx = similar(X, n, nlv)
    Ty = copy(Tx)
    Wx = similar(X, p, nlv)
    Wy = similar(X, q, nlv)
    Px = copy(Wx)
    Py = copy(Wy)
    TTx = similar(X, nlv)
    TTy = copy(TTx)
    tx   = similar(X, n)
    ty = copy(tx) 
    wx  = similar(X, p)
    wxtild = copy(wx)    
    wy  = similar(X, q)
    wytild = copy(wy)
    px   = copy(wx)
    py   = copy(wy)
    niter = zeros(nlv)
    # End
    @inbounds for a = 1:nlv
        tx .= X[:, 1]
        ty .= Y[:, 1]
        cont = true
        iter = 1
        wx .= rand(p)
        if tau == 0       
            invCx = inv(X' * X)
            invCy = inv(Y' * Y)
            #invCx = pinv(X' * X)
            #invCy = pinv(Y' * Y)
        else
            Ix = Diagonal(ones(p)) 
            Iy = Diagonal(ones(q)) 
            if tau == 1   
                invCx = Ix
                invCy = Iy
            else
                invCx = inv((1 - tau) * X' * X + tau * Ix)
                invCy = inv((1 - tau) * Y' * Y + tau * Iy)
            end
        end 
        while cont
            w0 = copy(wx)
            wxtild .= invCx * X' * ty / dot(ty, ty)
            wx .= wxtild / norm(wxtild)
            mul!(tx, X, wx)
            wytild .= invCy * Y' * tx / dot(tx, tx)
            wy .= wytild / norm(wytild)
            mul!(ty, Y, wy)
            dif = sum((wx .- w0).^2)
            iter = iter + 1
            if (dif < tol) || (iter > maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        ttx = dot(tx, tx)
        mul!(px, X', tx)
        px ./= ttx
        tty = dot(ty, ty)
        mul!(py, Y', ty)
        py ./= tty
        # Deflation
        X .-= tx * px'
        Y .-= ty * py'
        # Same as:
        #b = tx' * X / dot(tx, tx)
        #X .-= tx * b
        #b = ty' * Y / dot(ty, ty)
        #Y .-= ty * b
        # End         
        Tx[:, a] .= tx
        Ty[:, a] .= ty
        Wx[:, a] .= wx
        Wy[:, a] .= wy
        Px[:, a] .= px
        Py[:, a] .= py
        TTx[a] = ttx
        TTy[a] = tty
    end
    Tx .= (1 ./ sqrtw) .* Tx
    Ty .= (1 ./ sqrtw) .* Ty
    Rx = Wx * inv(Px' * Wx)
    Ry = Wy * inv(Py' * Wy)
    CcaWold(Tx, Ty, Px, Py, Rx, Ry, Wx, Wy, TTx, TTy, 
        bscales, xmeans, xscales, ymeans, yscales, weights, niter)
end

""" 
    transform(object::CcaWold, X, Y; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and (X, Y)-data.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute. If nothing, it is the maximum number
    from the fitted model.
""" 
function transform(object::CcaWold, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    Tx = cscale(X, object.xmeans, object.xscales) * vcol(object.Rx, 1:nlv)
    Ty = cscale(Y, object.ymeans, object.yscales) * vcol(object.Ry, 1:nlv)
    (Tx = Tx, Ty)
end

"""
    summary(object::CcaWold, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::CcaWold, X::Union{Vector, Matrix, DataFrame},
        Y::Union{Vector, Matrix, DataFrame})
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = cscale(X, object.xmeans, object.xscales)
    Y = cscale(Y, object.ymeans, object.yscales)
    ttx = object.TTx 
    tty = object.TTy 
    ## Explained variances
    # X
    sstot = fnorm(X, object.weights)^2
    tt_adj = vec(sum(object.Px.^2, dims = 1)) .* ttx
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    # Y
    sstot = fnorm(Y, object.weights)^2
    tt_adj = vec(sum(object.Py.^2, dims = 1)) .* tty
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
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


