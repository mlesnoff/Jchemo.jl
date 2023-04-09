
"""
    mbwcov(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "none", wcov = true, tau = 1,
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
    mbwcov!(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "none", wcov = true, tau = 1,
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
Multiblock weighted covariate analysis regression (MBWCov) (Mangana et al. 2021)
* `Xbl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `Y` : Y-data.
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscal` : Type of block scaling (`"none"`, `"frob"`). 
    See functions `blockscal`.
* `wcov` : Logical. If `true` (default), a MBWCov is done, else
    a MBPLSR is done.
* `tau` : Regularization parameter (∊ [0, 1]).     
* `tol` : Tolerance value for convergence.
* `maxit` : Maximum number of iterations.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` and 
    of `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

See Mangamana et al. 2021.

The regularization is implemented as a continuum. After block centering 
and scaling, the block covariances matrices (k = 1,...,K blocks) 
are computed as follows: 
* Ck = (1 - `tau`) * Xk' D Xk + `tau` * Ik
where D is the observation (row) metric. 
Value `tau` = 0 can generate unstability when inverting the covariance matrices. 
A better alternative is generally to use an epsilon value (e.g. `tau` = 1e-8) 
to get similar results as with pseudo-inverses.   

* When `tau` = 1, this is the PLS framework.
* When `tau` ~ 0, this is the redundancy analysis (RA) framework.

## References 
Tchandao Mangamana, E., Glèlè Kakaï, R., Qannari, E.M., 2021. A general strategy for setting 
up supervised methods of multiblock data analysis. Chemometrics and Intelligent Laboratory Systems 217, 
104388. https://doi.org/10.1016/j.chemolab.2021.104388

## Examples
```julia
using JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/ham.jld2") 
@load db dat
pnames(dat) 

X = dat.X
Y = dat.Y
y = dat.Y.c1
group = dat.group
listbl = [1:11, 12:19, 20:25]
Xbl = mblock(X, listbl)
# "New" = first two rows of Xbl 
Xbl_new = mblock(X[1:2, :], listbl)

bscal = "none"
nlv = 5
fm = mbwcov(Xbl, y; nlv = nlv, bscal = bscal) ;
pnames(fm)
fm.T
Jchemo.transform(fm, Xbl_new)
[y Jchemo.predict(fm, Xbl).pred]
Jchemo.predict(fm, Xbl_new).pred

summary(fm, Xbl)
```
"""
function mbwcov(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "none", wcov = true, tau = 1,
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
    nbl = length(Xbl)  
    zXbl = list(nbl, Matrix{Float64})
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    mbwcov!(zXbl, copy(ensure_mat(Y)), weights; nlv = nlv, 
        bscal = bscal, wcov = wcov, tau = tau, 
        tol = tol, maxit = maxit, scal = scal)
end

function mbwcov!(Xbl, Y::Matrix, weights = ones(nro(Xbl[1])); nlv,
        bscal = "none", wcov = true, tau = 1, 
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
    nbl = length(Xbl)
    n = nro(Xbl[1])
    q = nco(Y)
    weights = mweight(weights)
    sqrtw = sqrt.(weights)
    xmeans = list(nbl, Vector{Float64})
    xscales = list(nbl, Vector{Float64})
    p = fill(0, nbl)
    Threads.@threads for k = 1:nbl
        p[k] = nco(Xbl[k])
        xmeans[k] = colmean(Xbl[k], weights) 
        xscales[k] = ones(nco(Xbl[k]))
        if scal 
            xscales[k] = colstd(Xbl[k], weights)
            Xbl[k] .= cscale(Xbl[k], xmeans[k], xscales[k])
        else
            Xbl[k] .= center(Xbl[k], xmeans[k])
        end
    end
    ymeans = colmean(Y, weights)
    yscales = ones(q)
    if scal 
        yscales .= colstd(Y, weights)
        cscale!(Y, ymeans, yscales)
    else
        center!(Y, ymeans)
    end
    bscal == "none" ? bscales = ones(nbl) : nothing
    if bscal == "frob"
        res = blockscal_frob(Xbl, weights) 
        bscales = res.bscales
        Xbl = res.X
    end
    # Row metric
    @inbounds for k = 1:nbl
        Xbl[k] .= sqrtw .* Xbl[k]
    end
    Y .= sqrtw .* Y
    # Pre-allocation
    X = similar(Xbl[1], n, sum(p))
    Tbl = list(nbl, Matrix{Float64})
    for k = 1:nbl ; Tbl[k] = similar(Xbl[1], n, nlv) ; end
    Tb = list(nlv, Matrix{Float64})
    for a = 1:nlv ; Tb[a] = similar(Xbl[1], n, nbl) ; end
    Pbl = list(nbl, Matrix{Float64})
    for k = 1:nbl ; Pbl[k] = similar(Xbl[1], p[k], nlv) ; end
    Tx = similar(Xbl[1], n, nlv)
    Wx = similar(Xbl[1], sum(p), nlv)
    Wytild = similar(Xbl[1], q, nlv)
    Px = copy(Wx)
    tk  = similar(Xbl[1], n)
    tx = copy(tk)
    ty  = copy(tk)
    wx = similar(Xbl[1], sum(p))
    px = copy(wx)
    wy  = similar(Xbl[1], q)
    wytild = copy(wy)
    TTx = similar(Xbl[1], nlv)
    lb = ones(nbl, nlv)
    niter = zeros(nlv)
    if tau > 0 && tau < 1
        Ik = list(nbl, Array{Float64})
        @inbounds for k = 1:nbl
            Ik[k] = Diagonal(ones(p[k])) 
        end
    end
    # End
    @inbounds for a = 1:nlv
        ty = Y[:, 1]
        cont = true
        iter = 1
        while cont
            t0 = copy(ty)
            @inbounds for k = 1:nbl
                if tau == 0
                    tk .= Xbl[k] * inv(Xbl[k]' * Xbl[k]) *  Xbl[k]' * ty
                else
                    if tau == 1
                        tk .= Xbl[k] * Xbl[k]' * ty
                    else
                        tk .= Xbl[k] * inv((1 - tau) * Xbl[k]' * Xbl[k] + tau * Ik[k]) *  Xbl[k]' * ty
                    end
                end 
                Tb[a][:, k] .= tk
                Tbl[k][:, a] .= (1 ./ sqrtw) .* tk
                if wcov 
                    lbk = dot(ty, tk)
                    lb[k, a] = lbk
                end
            end
            TB = Tb[a] .* lb[:, a]'
            tx .= rowsum(TB) 
            wy .= Y' * tx 
            wy ./= norm(wy)
            ty .= Y * wy
            dif = sum((ty .- t0).^2)
            iter = iter + 1
            if (dif < tol) || (iter > maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        # For global
        ttx = dot(tx, tx)
        X .= reduce(hcat, Xbl)
        wx .= X' * ty / dot(ty, ty)    
        wx ./= norm(wx)
        mul!(px, X', tx)
        px ./= ttx
        wytild .= Y' * tx / ttx
        # End           
        Tx[:, a] .= tx   
        Wx[:, a] .= wx
        Px[:, a] .= px
        Wytild[:, a] .= wytild
        TTx[a] = ttx
        @inbounds for k = 1:nbl
            Xbl[k] .-= tx * tx' * Xbl[k] / ttx
            Y .-= tx * wytild'
        end
    end
    Tx .= (1 ./ sqrtw) .* Tx
    Rx = Wx * inv(Px' * Wx)
    MbplsWest(Tx, Px, Rx, Wx, Wytild, Tbl, Tb, Pbl, TTx,    
        bscales, xmeans, xscales, ymeans, yscales, weights, lb, niter)
end






