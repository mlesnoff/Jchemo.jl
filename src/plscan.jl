"""
    plscan(; kwargs...)
    plscan(X, Y; kwargs...)
    plscan(X, Y, weights::Weight; kwargs...)
    plscan!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Canonical partial least squares regression (Canonical PLS).
* `X` : First block of data.
* `Y` : Second block of data.
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs = scores) to compute.
* `bscal` : Type of block scaling. Possible values are: `:none`, `:frob`. See functions `blockscal`.
* `scal` : Boolean. If `true`, each column of blocks `X` and `Y` is scaled by its uncorrected 
    standard deviation (before the block scaling).

Canonical PLS with the Nipals algorithm (Wold 1984, Tenenhaus 1998 chap.11), referred to as PLS-W2A 
(i.e. Wold PLS mode A) in Wegelin 2000. The two blocks `X` and `Y` play a symmetric role.  After each 
step of scores computation, X and Y are deflated by the x- and y-scores, respectively. 

Function `summary` returns:
* `cortx2ty`: Correlations between the X- and Y-LVs
and for block `X`: 
* `explvarx` : Proportion of the block inertia (squared Frobenious norm) explained by the block LVs (`Tx`).
* `rvx2tx` : RV coefficients between the block and the block LVs.
* `rdx2tx` : Rd coefficients between the block and the block LVs.
* `corx2tx` : Correlation between the block variables and the block LVs. The same is returned for block `Y`.  

## References
Tenenhaus, M., 1998. La régression PLS: théorie et pratique. Editions Technip, Paris.

Wegelin, J.A., 2000. A Survey of Partial Least Squares (PLS) Methods, with Emphasis on 
the Two-Block Case (No. 371). University of Washington, Seattle, Washington, USA.

Wold, S., Ruhe, A., Wold, H., Dunn, III, W.J., 1984. The Collinearity Problem in Linear 
Regression. The Partial Least Squares (PLS) Approach to Generalized Inverses. 
SIAM Journal on Scientific and Statistical Computing 5, 735–743. https://doi.org/10.1137/0905052

## Examples
```julia
using Jchemo, JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "linnerud.jld2") 
@load db dat
@names dat
X = dat.X
Y = dat.Y
n, p = size(X)
q = nco(Y)

nlv = 2
bscal = :frob
model = plscan(; nlv, bscal)
fit!(model, X, Y)
@names model
@names model.fitm

fitm = model.fitm
@head fitm.Tx
@head transfbl(model, X, Y).Tx

@head fitm.Ty
@head transfbl(model, X, Y).Ty

res = summary(model, X, Y) ;
@names res
res.explvarx
res.explvary
res.cortx2ty
res.rvx2tx
res.rvy2ty
res.rdx2tx
res.rdy2ty
res.corx2tx 
res.cory2ty 
```
"""
plscan(; kwargs...) = JchemoModel(plscan, nothing, kwargs)

function plscan(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    n = nro(X)
    weights = mweight(ones(Q, n))
    plscan(X, Y, weights; kwargs...)
end

function plscan(X, Y, weights::Weight; kwargs...)
    plscan!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function plscan!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParPls2bl, kwargs).par 
    @assert in([:none, :frob])(par.bscal) "Wrong value for argument 'bscal'."
    Q = eltype(X)
    n, p = size(X)
    q = nco(Y)
    nlv = min(par.nlv, p, q)
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
    ## Pre-allocation
    XtY = similar(X, p, q)
    Tx = similar(X, n, nlv)
    Ty = copy(Tx)
    Wx = similar(X, p, nlv)
    Wy = similar(X, q, nlv) 
    Vx = copy(Wx)
    Vy = copy(Wy)
    TTx = similar(X, nlv)
    TTy = copy(TTx)
    tx = similar(X, n)
    ty = copy(tx)
    dtx  = copy(tx)
    dty = copy(tx)   
    wx = similar(X, p)
    wy = similar(X, q)
    vx = copy(wx)
    vy = copy(wy)
    delta = copy(TTx)
    # End
    @inbounds for a = 1:nlv
        XtY .= X' * fweight(Y, weights.w)
        U, d, V = svd!(XtY) 
        delta[a] = d[1]
        # X
        wx .= U[:, 1]
        mul!(tx, X, wx)
        dtx .= weights.w .* tx
        ttx = dot(tx, dtx)
        mul!(vx, X', dtx)
        vx ./= ttx
        # Y
        wy .= V[:, 1]
        # Same as:                        
        # mul!(wy, Y', dtx)
        # wy ./= normv(wy)
        # End
        mul!(ty, Y, wy)
        dty .= weights.w .* ty
        tty = dot(ty, dty)
        mul!(vy, Y', dty)
        vy ./= tty
        # Deflation
        X .-= tx * vx'
        Y .-= ty * vy'
        # End
        Vx[:, a] .= vx
        Vy[:, a] .= vy
        Tx[:, a] .= tx
        Ty[:, a] .= ty
        Wx[:, a] .= wx
        Wy[:, a] .= wy
        TTx[a] = ttx
        TTy[a] = tty
     end
     Rx = Wx * inv(Vx' * Wx)
     Ry = Wy * inv(Vy' * Wy)
     Plscan(Tx, Ty, Vx, Vy, Rx, Ry, Wx, Wy, TTx, TTy, delta, bscales, xmeans, xscales, 
         ymeans, yscales, weights, par)
end

""" 
    transfbl(object::Plscan, X, Y; nlv = nothing)
Compute latent variables (LVs = scores) from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transfbl(object::Plscan, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    X = fcscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = fcscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    Tx = X * vcol(object.Rx, 1:nlv)
    Ty = Y * vcol(object.Ry, 1:nlv)
    (Tx = Tx, Ty)
end

"""
    summary(object::Plscan, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::Plscan, X, Y)
    Q = eltype(X[1, 1])
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = fcscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = fcscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    ttx = object.TTx 
    tty = object.TTy 
    ## Block X
    ss = frob2(X, object.weights)
    tt_adj = (colnorm(object.Vx).^2) .* ttx  
    pvar = tt_adj / ss
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## Block Y
    ss = frob2(Y, object.weights)
    tt_adj = (colnorm(object.Vy).^2) .* tty  
    pvar = tt_adj / ss
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvary = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## Correlation between X- and Y-block LVs
    z = diag(corm(object.Tx, object.Ty, object.weights))
    cortx2ty = DataFrame(lv = 1:nlv, cor = z)
    ## RV(X, tx) and RV(Y, ty)
    nam = string.("lv", 1:nlv)
    z = zeros(Q, 1, nlv)
    for a = 1:nlv
        z[1, a] = rv(X, object.Tx[:, a], object.weights) 
    end
    rvx2tx = DataFrame(z, nam)
    for a = 1:nlv
        z[1, a] = rv(Y, object.Ty[:, a], object.weights) 
    end
    rvy2ty = DataFrame(z, nam)
    ## Redundancies (Average correlations) Rd(X, tx) and Rd(Y, ty)
    z[1, :] = rd(X, object.Tx, object.weights) 
    rdx2tx = DataFrame(z, nam)
    z[1, :] = rd(Y, object.Ty, object.weights) 
    rdy2ty = DataFrame(z, nam)
    ## Correlation between block variables and their block LVs
    z = corm(X, object.Tx, object.weights)
    corx2tx = DataFrame(z, string.("lv", 1:nlv))
    z = corm(Y, object.Ty, object.weights)
    cory2ty = DataFrame(z, string.("lv", 1:nlv))
    ## End
    (explvarx = explvarx, explvary, cortx2ty, rvx2tx, rvy2ty, rdx2tx, rdy2ty, 
        corx2tx, cory2ty)
end

