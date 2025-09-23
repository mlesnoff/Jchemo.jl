"""
    ccawold(; kwargs...)
    ccawold(X, Y; kwargs...)
    ccawold(X, Y, weights::Weight; kwargs...)
    ccawold!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Canonical correlation analysis (CCA, RCCA) - Wold Nipals algorithm.
* `X` : First block of data.
* `Y` : Second block of data.
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs; = scores) to compute.
* `bscal` : Type of block scaling. Possible values are:`:none`, `:frob`. See functions `blockscal`.
* `tau` : Regularization parameter (∊ [0, 1]).
* `tol` : Tolerance value for convergence (Nipals).
* `maxit` : Maximum number of iterations (Nipals).
* `scal` : Boolean. If `true`, each column of blocks `X` and `Y` is scaled by its uncorrected standard 
    deviation (before the block scaling).

This function implements the Nipals ccawold algorithm presented by Tenenhaus 1998 p.204 (related to Wold et al. 1984). 

In this implementation, after each step of LVs computation, X and Y are deflated relatively to their respective scores 
(tx and ty). 

A continuum regularization is available (parameter `tau`). After block centering and scaling, the covariances matrices 
are computed as follows: 
* Cx = (1 - `tau`) * X'DX + `tau` * Ix
* Cy = (1 - `tau`) * Y'DY + `tau` * Iy
where D is the observation (row) metric. Value `tau` = 0 can generate unstability when inverting the covariance 
matrices. Often, a better alternative is to use an epsilon value (e.g. `tau` = 1e-8) to get similar results as with 
pseudo-inverses.   

The normed scores returned by the function are expected (using uniform `weights`) to be the same as those returned 
by function `rgcca` of the R package `RGCCA` (Tenenhaus & Guillemot 2017, Tenenhaus et al. 2017). 

See function `plscan` for the details on the `summary` outputs.

## References
Tenenhaus, A., Guillemot, V. 2017. RGCCA: Regularized and Sparse Generalized Canonical Correlation Analysis for 
Multiblock Data Multiblock data analysis.https://cran.r-project.org/web/packages/RGCCA/index.html 

Tenenhaus, M., 1998. La régression PLS: théorie et pratique. Editions Technip, Paris.

Tenenhaus, M., Tenenhaus, A., Groenen, P.J.F., 2017. Regularized Generalized Canonical Correlation Analysis: A Framework 
for Sequential Multiblock Component Methods. Psychometrika 82, 737–777. https://doi.org/10.1007/s11336-017-9573-x

Wold, S., Ruhe, A., Wold, H., Dunn, III, W.J., 1984. The Collinearity Problem in Linear Regression. The Partial Least 
Squares (PLS) Approach to Generalized Inverses. SIAM Journal on Scientific and Statistical Computing 5, 735–743. 
https://doi.org/10.1137/0905052

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
bscal = :frob ; tau = 1e-4
model = ccawold(; nlv, bscal, tau, tol = 1e-10)
fit!(model, X, Y)
@names model
@names model.fitm

@head model.fitm.Tx
@head transfbl(model, X, Y).Tx

@head model.fitm.Ty
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
ccawold(; kwargs...) = JchemoModel(ccawold, nothing, kwargs)

function ccawold(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    n = nro(X)
    weights = mweight(ones(Q, n))
    ccawold(X, Y, weights; kwargs...)
end

function ccawold(X, Y, weights::Weight; kwargs...)
    ccawold!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function ccawold!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParCcawold, kwargs).par 
    @assert in([:none, :frob])(par.bscal) "Wrong value for argument 'bscal'."
    @assert 0 <= par.tau <= 1 "tau must be in [0, 1]"
    Q = eltype(X)
    n, p = size(X)
    q = nco(Y)
    nlv = min(par.nlv, p, q)
    tau = convert(Q, par.tau) 
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
    sqrtw = sqrt.(weights.w)
    invsqrtw = 1 ./ sqrtw
    fweight!(X, sqrtw)
    fweight!(Y, sqrtw)
    ## Pre-allocation
    Tx = similar(X, n, nlv)
    Ty = copy(Tx)
    Wx = similar(X, p, nlv)
    Wy = similar(X, q, nlv)
    Vx = copy(Wx)
    Vy = copy(Wy)
    TTx = similar(X, nlv)
    TTy = copy(TTx)
    tx   = similar(X, n)
    ty = copy(tx) 
    wx  = similar(X, p)
    wxtild = copy(wx)    
    wy  = similar(X, q)
    wytild = copy(wy)
    vx   = copy(wx)
    vy   = copy(wy)
    niter = zeros(nlv)
    Ix = Diagonal(ones(Q, p)) 
    Iy = Diagonal(ones(Q, q)) 
    # End
    @inbounds for a = 1:nlv
        tx .= X[:, 1]
        ty .= Y[:, 1]
        cont = true
        iter = 1
        wx .= convert.(Q, rand(p))
        ## invCx, invCy
        if tau == 0       
            invCx = inv(X' * X)
            invCy = inv(Y' * Y)
        else
            if tau == 1   
                invCx = copy(Ix)
                invCy = copy(Iy)
            else
                invCx = inv((1 - tau) * X' * X + tau * Ix)
                invCy = inv((1 - tau) * Y' * Y + tau * Iy)
            end
        end 
        ## End
        ttx = 0
        tty = 0
        while cont
            w0 = copy(wx)
            tty = dot(ty, ty)
            wxtild .= invCx * X' * ty / tty
            wx .= wxtild / normv(wxtild)
            mul!(tx, X, wx)
            ttx = dot(tx, tx)
            wytild .= invCy * Y' * tx / ttx
            wy .= wytild / normv(wytild)
            mul!(ty, Y, wy)
            dif = sum((wx .- w0).^2)
            iter = iter + 1
            if (dif < par.tol) || (iter > par.maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        mul!(vx, X', tx)
        vx ./= ttx
        mul!(vy, Y', ty)
        vy ./= tty
        # Deflation
        X .-= tx * vx'
        Y .-= ty * vy'
        # Same as:
        #b = tx' * X / ttx
        #X .-= tx * b
        #b = ty' * Y / tty
        #Y .-= ty * b
        # End         
        Tx[:, a] .= tx
        Ty[:, a] .= ty
        Wx[:, a] .= wx
        Wy[:, a] .= wy
        Vx[:, a] .= vx
        Vy[:, a] .= vy
        TTx[a] = ttx
        TTy[a] = tty
    end
    fweight!(Tx, invsqrtw)
    fweight!(Ty, invsqrtw)
    Rx = Wx * inv(Vx' * Wx)
    Ry = Wy * inv(Vy' * Wy)
    Ccawold(Tx, Ty, Vx, Vy, Rx, Ry, Wx, Wy, TTx, TTy, bscales, xmeans, xscales, ymeans, yscales, 
        weights, niter, par)
end

""" 
    transfbl(object::Ccawold, X, Y; nlv = nothing)
Compute latent variables (LVs; = scores) from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transfbl(object::Ccawold, X, Y; nlv = nothing)
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
    summary(object::Ccawold, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::Ccawold, X, Y)
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
    (explvarx = explvarx, explvary, cortx2ty, rvx2tx, rvy2ty, rdx2tx, rdy2ty, corx2tx, cory2ty)
end


