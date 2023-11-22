"""
    plscan(X, Y, weights = ones(nro(X)); nlv,
        bscal = :none, scal::Bool = false)
    plscan!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        bscal = :none, scal::Bool = false)
Canonical partial least squares regression (Canonical PLS)
* `X` : First block (matrix) of data.
* `Y` : Second block (matrix) of data.
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling (`:none`, `:frob`). 
    See functions `blockscal`.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

Canonical PLS with the Nipals algorithm (Wold 1984, Tenenhaus 1998 chap.11),
referred to as PLS-W2A (i.e. Wold PLS mode A) in Wegelin 2000.
The two blocks `X` and `X` play a symmetric role.  
After each step of scores computation, X and Y are deflated by the x- and 
y-scores, respectively. 

## References
Tenenhaus, M., 1998. La régression PLS: théorie et pratique. Editions Technip, Paris.

Wegelin, J.A., 2000. A Survey of Partial Least Squares (PLS) Methods, with Emphasis 
on the Two-Block Case (No. 371). University of Washington, Seattle, Washington, USA.

Wold, S., Ruhe, A., Wold, H., Dunn, III, W.J., 1984. The Collinearity Problem in Linear 
Regression. The Partial Least Squares (PLS) Approach to Generalized Inverses. 
SIAM Journal on Scientific and Statistical Computing 5, 735–743. 
https://doi.org/10.1137/0905052

## Examples
```julia
using JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/linnerud.jld2") 
@load db dat
pnames(dat)
X = dat.X 
Y = dat.Y

fm = plscan(X, Y; nlv = 3)
pnames(fm)

fm.Tx
transform(fm, X, Y).Tx
scale(fm.Tx, colnorm(fm.Tx))

res = summary(fm, X, Y)
pnames(res)
```
"""
function plscan(X, Y; par = Par())
    Q = eltype(X[1, 1])
    n = nro(X)
    weights = mweight(ones(Q, n))
    plscan(X, Y, weights; par)
end

function plscan(X, Y, weights::Weight; par = Par())
    plscan!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; par)
end

function plscan!(X::Matrix, Y::Matrix, weights::Weight; 
        par = Par())
    @assert in([:none, :frob])(par.bscal) "Wrong value for argument 'bscal'."
    Q = eltype(X)
    n, p = size(X)
    q = nco(Y)
    nlv = min(par.nlv, p, q)
    D = Diagonal(weights.w)
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
    par.bscal == :none ? bscales = ones(2) : nothing
    if par.bscal == :frob
        normx = frob(X, weights)
        normy = frob(Y, weights)
        X ./= normx
        Y ./= normy
        bscales = [normx; normy]
    end
    # Pre-allocation
    XtY = similar(X, p, q)
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
    dtx  = copy(tx)
    dty = copy(tx)   
    wx  = similar(X, p)
    wy  = similar(X, q)
    px   = copy(wx)
    py   = copy(wy)
    delta = copy(TTx)
    # End
    @inbounds for a = 1:nlv
        XtY .= X' * D * Y
        U, d, V = svd!(XtY) 
        delta[a] = d[1]
        # X
        wx .= U[:, 1]
        mul!(tx, X, wx)
        dtx .= weights.w .* tx
        ttx = dot(tx, dtx)
        mul!(px, X', dtx)
        px ./= ttx
        # Y
        wy .= V[:, 1]
        # Same as:                        
        # mul!(wy, Y', dtx)
        # wy ./= norm(wy)
        # End
        mul!(ty, Y, wy)
        dty .= weights.w .* ty
        tty = dot(ty, dty)
        mul!(py, Y', dty)
        py ./= tty
        # Deflation
        X .-= tx * px'
        Y .-= ty * py'
        # End
        Px[:, a] .= px
        Py[:, a] .= py
        Tx[:, a] .= tx
        Ty[:, a] .= ty
        Wx[:, a] .= wx
        Wy[:, a] .= wy
        TTx[a] = ttx
        TTy[a] = tty
     end
     Rx = Wx * inv(Px' * Wx)
     Ry = Wy * inv(Py' * Wy)
     PlsCan(Tx, Ty, Px, Py, Rx, Ry, Wx, Wy, TTx, TTy, delta, 
         bscales, xmeans, xscales, ymeans, yscales, weights)
end

""" 
    transform(object::PlsCan, X, Y; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and (X, Y)-data.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute. If nothing, it is the maximum number
    from the fitted model.
""" 
function transform(object::PlsCan, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    Tx = cscale(X, object.xmeans, object.xscales) * vcol(object.Rx, 1:nlv)
    Ty = cscale(Y, object.ymeans, object.yscales) * vcol(object.Ry, 1:nlv)
    (Tx = Tx, Ty)
end

"""
    summary(object::PlsCan, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::PlsCan, X,
        Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = cscale(X, object.xmeans, object.xscales)
    Y = cscale(Y, object.ymeans, object.yscales)
    ttx = object.TTx 
    tty = object.TTy 
    # X
    sstot = frob(X, object.weights)^2
    tt_adj = colsum(object.Px.^2) .* ttx
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, 
        cumpvar = cumpvar)
    # Y
    sstot = frob(Y, object.weights)^2
    tt_adj = colsum(object.Py.^2) .* tty
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvary = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, 
        cumpvar = cumpvar)
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

