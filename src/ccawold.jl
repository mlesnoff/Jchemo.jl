"""
    ccawold(X, Y; kwargs...)
    ccawold(X, Y, weights::Weight; kwargs...)
    ccawold!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Canonical correlation analysis (CCA, RCCA) - Wold 
    Nipals algorithm.
* `X` : First block of data.
* `Y` : Second block of data.
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling. Possible values are:
    `:none`, `:frob`. See functions `blockscal`.
* `tau` : Regularization parameter (∊ [0, 1]).
* `tol` : Tolerance value for convergence (Nipals).
* `maxit` : Maximum number of iterations (Nipals).
* `scal` : Boolean. If `true`, each column of blocks `X` 
    and `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

This function implements the Nipals ccawold algorithm 
presented by Tenenhaus 1998 p.204 (related to Wold et al. 1984). 

In this implementation, after each step of LVs computation, 
X and Y are deflated relatively to their respective scores 
(tx and ty). 

A continuum regularization is available (parameter `tau`). 
After block centering and scaling, the covariances matrices 
are computed as follows: 
* Cx = (1 - `tau`) * X'DX + `tau` * Ix
* Cy = (1 - `tau`) * Y'DY + `tau` * Iy
where D is the observation (row) metric. 
Value `tau` = 0 can generate unstability when inverting 
the covariance matrices. Often, a better alternative is 
to use an epsilon value (e.g. `tau` = 1e-8) to get similar 
results as with pseudo-inverses.   

The normed scores returned by the function are expected 
(using uniform `weights`) to be the same as those 
returned by function `rgcca` of the R package `RGCCA` 
(Tenenhaus & Guillemot 2017, Tenenhaus et al. 2017). 

## References
Tenenhaus, A., Guillemot, V. 2017. RGCCA: Regularized and 
Sparse Generalized Canonical Correlation Analysis for 
Multiblock Data Multiblock data analysis.
https://cran.r-project.org/web/packages/RGCCA/index.html 

Tenenhaus, M., 1998. La régression PLS: théorie et 
pratique. Editions Technip, Paris.

Tenenhaus, M., Tenenhaus, A., Groenen, P.J.F., 2017. 
Regularized Generalized Canonical Correlation Analysis: 
A Framework for Sequential Multiblock Component Methods. 
Psychometrika 82, 737–777. 
https://doi.org/10.1007/s11336-017-9573-x

Wold, S., Ruhe, A., Wold, H., Dunn, III, W.J., 1984. 
The Collinearity Problem in Linear Regression. The Partial 
Least Squares (PLS) Approach to Generalized Inverses. 
SIAM Journal on Scientific and Statistical Computing 5, 
735–743. https://doi.org/10.1137/0905052

## Examples
```julia
using Jchemo, JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "linnerud.jld2") 
@load db dat
pnames(dat)
X = dat.X
Y = dat.Y
n, p = size(X)
q = nco(Y)

nlv = 2
bscal = :frob ; tau = 1e-4
model = ccawold; nlv, bscal, tau, tol = 1e-10)
fit!(model, X, Y)
pnames(model)
pnames(model.fm)

@head model.fm.Tx
@head transfbl(model, X, Y).Tx

@head model.fm.Ty
@head transfbl(model, X, Y).Ty

res = summary(model, X, Y) ;
pnames(res)
res.explvarx
res.explvary
res.cort2t 
res.rdx
res.rdy
res.corx2t 
res.cory2t 
```
"""
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
    ## Pre-allocation
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
        wx .= convert.(Q, rand(p))
        if tau == 0       
            invCx = inv(X' * X)
            invCy = inv(Y' * Y)
        else
            Ix = Diagonal(ones(Q, p)) 
            Iy = Diagonal(ones(Q, q)) 
            if tau == 1   
                invCx = Ix
                invCy = Iy
            else
                invCx = inv((1 - tau) * X' * X + tau * Ix)
                invCy = inv((1 - tau) * Y' * Y + tau * Iy)
            end
        end 
        ttx = 0
        tty = 0
        while cont
            w0 = copy(wx)
            tty = dot(ty, ty)
            wxtild .= invCx * X' * ty / tty
            wx .= wxtild / norm(wxtild)
            mul!(tx, X, wx)
            ttx = dot(tx, tx)
            wytild .= invCy * Y' * tx / ttx
            wy .= wytild / norm(wytild)
            mul!(ty, Y, wy)
            dif = sum((wx .- w0).^2)
            iter = iter + 1
            if (dif < par.tol) || (iter > par.maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        mul!(px, X', tx)
        px ./= ttx
        mul!(py, Y', ty)
        py ./= tty
        # Deflation
        X .-= tx * px'
        Y .-= ty * py'
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
        Px[:, a] .= px
        Py[:, a] .= py
        TTx[a] = ttx
        TTy[a] = tty
    end
    Tx .= (1 ./ sqrtw) .* Tx
    Ty .= (1 ./ sqrtw) .* Ty
    Rx = Wx * inv(Px' * Wx)
    Ry = Wy * inv(Py' * Wy)
    Ccawold(Tx, Ty, Px, Py, Rx, Ry, Wx, Wy, TTx, TTy, bscales, xmeans, xscales, 
        ymeans, yscales, weights, niter, par)
end

""" 
    transfbl(object::Ccawold, X, Y; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model.
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
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = fcscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = fcscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    # X
    tt = object.TTx 
    sstot = frob(X, object.weights)^2
    tt_adj = colsum(object.Px.^2) .* tt
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, 
        cumpvar = cumpvar)
    # Y
    tt = object.TTy 
    sstot = frob(Y, object.weights)^2
    tt_adj = colsum(object.Py.^2) .* tt
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvary = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, 
        cumpvar = cumpvar)
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
    (explvarx = explvarx, explvary, cort2t, rdx, rdy, corx2t, cory2t)
end


