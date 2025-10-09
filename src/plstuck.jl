"""
    plstuck(; kwargs...)
    plstuck(X, Y; kwargs...)
    plstuck(X, Y, weights::Weight; kwargs...)
    plstuck!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Tucker's inter-battery method of factor analysis
* `X` : First block of data.
* `Y` : Second block of data.
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs; = scores) to compute.
* `bscal` : Type of block scaling. Possible values are:
    `:none`, `:frob`. See functions `blockscal`.
* `scal` : Boolean. If `true`, each column of blocks `X` and `Y` is scaled by its uncorrected 
    standard deviation (before the block scaling).

Inter-battery method of factor analysis (Tucker 1958, Tenenhaus 1998 chap.3). The two blocks 
`X` and `X` play a symmetric role.  This method is referred to as PLS-SVD in Wegelin 2000. The method 
factorizes the covariance matrix X'Y by SVD. 

See function `plscan` for the details on the `summary` outputs.

## References
Tenenhaus, M., 1998. La régression PLS: théorie et pratique. Editions Technip, Paris.

Tishler, A., Lipovetsky, S., 2000. Modelling and forecasting with robust canonical 
analysis: method and application. Computers & Operations Research 27, 217–232. 
https://doi.org/10.1016/S0305-0548(99)00014-3

Tucker, L.R., 1958. An inter-battery method of factor analysis. Psychometrika 23, 111–136.
https://doi.org/10.1007/BF02289009

Wegelin, J.A., 2000. A Survey of Partial Least Squares (PLS) Methods, with Emphasis 
on the Two-Block Case (No. 371). University of Washington, Seattle, Washington, USA.

## Examples
```julia
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/linnerud.jld2") 
@load db dat
@names dat
X = dat.X 
Y = dat.Y

model = plstuck(nlv = 3)
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
plstuck(; kwargs...) = JchemoModel(plstuck, nothing, kwargs)

function plstuck(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    n = nro(X)
    weights = mweight(ones(Q, n))
    plstuck(X, Y, weights; kwargs...)
end

function plstuck(X, Y, weights::Weight; kwargs...)
    plstuck!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function plstuck!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParPls2bl, kwargs).par 
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
    XtY = X' * rweight(Y, weights.w)
    U, delta, V = svd(XtY)
    delta = delta[1:nlv]
    Wx = U[:, 1:nlv]
    Wy = V[:, 1:nlv]
    Tx = X * Wx
    Ty = Y * Wy
    TTx = colnorm(Tx, weights).^2
    TTy = colnorm(Ty, weights).^2
    Plstuck(Tx, Ty, Wx, Wy, TTx, TTy, delta, bscales, xmeans, xscales, ymeans, yscales, 
        weights, par)
end

""" 
    transfbl(object::Plstuck, X, Y; nlv = nothing)
Compute latent variables (LVs; = scores) from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transfbl(object::Plstuck, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = object.par.nlv
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    X = fcscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = fcscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    Tx = X * vcol(object.Wx, 1:nlv)
    Ty = Y * vcol(object.Wy, 1:nlv)
    (Tx = Tx, Ty)
end

"""
    summary(object::Plstuck, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::Plstuck, X, Y)
    Q = eltype(X[1, 1])
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = fcscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = fcscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    ## Block X
    tt = object.TTx
    ss = frob2(X, object.weights)
    pvar = tt / ss
    cumpvar = cumsum(pvar) 
    xvar = tt / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## Block Y
    tt = object.TTy
    ss = frob2(Y, object.weights)
    pvar = tt / ss
    cumpvar = cumsum(pvar)
    xvar = tt / n    
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
    (explvarx = explvarx, explvary, cortx2ty, rvx2tx, rvy2ty, rdx2tx, rdy2ty, corx2tx, cory2ty)
end
