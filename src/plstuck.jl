"""
    plstuck(X, Y, weights = ones(nro(X)); nlv,
        bscal = :none, scal::Bool = false)
    plstuck!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        bscal = :none, scal::Bool = false)
Tucker's inter-battery method of factor analysis
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

Inter-battery method of factor analysis (Tucker 1958, Tenenhaus 1998 chap.3), 
The two blocks `X` and `X` play a symmetric role.  This method is referred to 
as PLS-SVD in Wegelin 2000. The basis of the method is to factorize the covariance 
matrix X'Y by SVD. 

## References

Tenenhaus, M., 1998. La régression PLS: théorie et pratique. Editions Technip, Paris.

Tishler, A., Lipovetsky, S., 2000. Modelling and forecasting with robust 
canonical analysis: method and application. Computers & Operations Research 27, 
217–232. https://doi.org/10.1016/S0305-0548(99)00014-3

Tucker, L.R., 1958. An inter-battery method of factor analysis. Psychometrika 23, 111–136.
https://doi.org/10.1007/BF02289009

Wegelin, J.A., 2000. A Survey of Partial Least Squares (PLS) Methods, with Emphasis 
on the Two-Block Case (No. 371). University of Washington, Seattle, Washington, USA.

## Examples
```julia
using JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/linnerud.jld2") 
@load db dat
pnames(dat)
X = dat.X 
Y = dat.Y

fm = plstuck(X, Y; nlv = 3)
pnames(fm)

fm.Tx
transform(fm, X, Y).Tx
scale(fm.Tx, colnorm(fm.Tx))

res = summary(fm, X, Y)
pnames(res)
```
"""
function plstuck(X, Y; par = Par())
    Q = eltype(X[1, 1])
    n = nro(X)
    weights = mweight(ones(Q, n))
    plstuck(X, Y, weights; par)
end

function plstuck(X, Y, weights::Weight; par = Par())
    plstuck!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; par)
end

function plstuck!(X::Matrix, Y::Matrix, weights::Weight; 
        par = Par())
    @assert in([:none, :frob])(par.bscal) "Wrong value for argument 'bscal'."
    Q = eltype(X)
    p = nco(X)
    q = nco(Y)
    nlv = min(par.nlv, p, q)
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
    D = Diagonal(weights.w)
    XtY = X' * D * Y
    U, delta, V = svd(XtY)
    delta = delta[1:nlv]
    Wx = U[:, 1:nlv]
    Wy = V[:, 1:nlv]
    Tx = X * Wx
    Ty = Y * Wy
    TTx = colsum(D * Tx .* Tx)
    TTy = colsum(D * Ty .* Ty)
    PlsTuck(Tx, Ty, Wx, Wy, TTx, TTy, delta, bscales, xmeans, xscales, 
        ymeans, yscales, weights)
end

""" 
    transform(object::PlsSVd, X, Y; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and (X, Y)-data.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute. If nothing, it is the maximum number
    from the fitted model.
""" 
function transform(object::PlsTuck, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    Tx = cscale(X, object.xmeans, object.xscales) * vcol(object.Wx, 1:nlv)
    Ty = cscale(Y, object.ymeans, object.yscales) * vcol(object.Wy, 1:nlv)
    (Tx = Tx, Ty)
end

"""
    summary(object::PlsTuck, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::PlsTuck, X::Union{Vector, Matrix, DataFrame},
        Y::Union{Vector, Matrix, DataFrame})
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = cscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = cscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    # X
    tt = object.TTx
    sstot = frob(X, object.weights)^2
    pvar = tt / sstot
    cumpvar = cumsum(pvar) 
    xvar = tt / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, 
        cumpvar = cumpvar)
    # Y
    tt = object.TTy
    sstot = frob(Y, object.weights)^2
    pvar = tt / sstot
    cumpvar = cumsum(pvar)
    xvar = tt / n    
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
