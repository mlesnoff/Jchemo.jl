"""
    mbunif(Xbl, weights = ones(nro(Xbl[1])); nlv,
        bscal = "none", tau = 1e-8, wcov = false, deflat = "global", 
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
    mbunif!(Xbl, weights = ones(nro(Xbl[1])); nlv,
        bscal = "none", tau = 1e-8, wcov = false, deflat = "global", 
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
Unified multiblock analysis of Mangana et al. 2019.
* `Xbl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling (`"none"`, `"frob"`, `"mfa"`). 
    See functions `blockscal`.
* `tau` : Regularization parameter (∊ [0, 1]).
* `wcov` : If `false`, the global score is proportionnal to the sum of 
    the block scores. If `true`, it is proportionnal to the weighted 
    sum of the block scores (with weights proportionnal to the covariances
    between the block scores and the global scores).
* `deflat` : Possible values are "global (deflation to the global scores)
    or "can" (deflation to the block scores).
* `tol` : Tolerance value for convergence.
* `maxit` : Maximum number of iterations.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

See Mangana et al. 2019.
The regularization parameter `tau` = "1 - gamma" in 
Managana et al. 2019 section 2.1.3.

Value `tau` = 0 can generate unstability when inverting the covariance matrices. 
A better alternative is generally to use an epsilon value (e.g. `tau` = 1e-8) 
to get similar results as with pseudo-inverses.  

## References
Mangamana, E.T., Cariou, V., Vigneau, E., Glèlè Kakaï, R.L., Qannari, E.M., 2019. 
Unsupervised multiblock data analysis: A unified approach and extensions. Chemometrics and 
Intelligent Laboratory Systems 194, 103856. https://doi.org/10.1016/j.chemolab.2019.103856

## Examples
```julia
using JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "linnerud.jld2") 
@load db dat
pnames(dat)
Xbl = [dat.X, dat.Y]

tau = 1e-8
fm = mbunif(Xbl; nlv = 3, tau = tau)
pnames(fm)

fm.T
transform(fm, Xbl).T

res = summary(fm, Xbl)
pnames(res)

## MBPCA
fm = mbunif(Xbl; nlv = 3,
    tau = 1, wcov = false, deflat = "global") ;

## ComDim
fm = mbunif(Xbl; nlv = 3,
    tau = 1, wcov = true, deflat = "global") ;
```
"""
function mbunif(Xbl, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "none", tau = 1e-8, wcov = false, deflat = "global",
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
    nbl = length(Xbl)  
    zXbl = list(nbl, Matrix{Float64})
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    mbunif!(zXbl, weights; nlv = nlv, 
        bscal = bscal, tau = tau, wcov = wcov, deflat = deflat, 
        tol = tol, maxit = maxit, scal = scal)
end

function mbunif!(Xbl, weights = ones(nro(Xbl[1])); nlv,
        bscal = "none", tau = 1e-8, wcov = false, deflat = "global", 
        tol = sqrt(eps(1.)), maxit = 200, scal = false)
    @assert tau >= 0 && tau <= 1 "tau must be in [0, 1]"
    nbl = length(Xbl)
    n = nro(Xbl[1])
    weights = mweight(weights)
    sqrtw = sqrt.(weights)
    xmeans = list(nbl, Vector{Float64})
    xscales = list(nbl, Vector{Float64})
    p = fill(0, nbl)
    @inbounds for k = 1:nbl
        p[k] = nco(Xbl[k])
        xmeans[k] = colmean(Xbl[k], weights) 
        xscales[k] = ones(nco(Xbl[k]))
        if scal 
            xscales[k] = colstd(Xbl[k], weights)
            Xbl[k] = cscale(Xbl[k], xmeans[k], xscales[k])
        else
            Xbl[k] = center(Xbl[k], xmeans[k])
        end
    end
    bscal == "none" ? bscales = ones(nbl) : nothing
    if bscal == "frob"
        res = blockscal_frob(Xbl, weights) 
        bscales = res.bscales
        Xbl = res.X
    end
    # Row metric
    sqrtw = sqrt.(weights)
    @inbounds for k = 1:nbl
        Xbl[k] .= sqrtw .* Xbl[k]
    end
    # Pre-allocation
    u = similar(Xbl[1], n)
    q = copy(u)
    qk = copy(u)
    U = similar(Xbl[1], n, nlv)
    Tbl = list(nbl, Matrix{Float64})
    for k = 1:nbl ; Tbl[k] = similar(Xbl[1], n, nlv) ; end
    Qbl = list(nbl, Matrix{Float64})
    for k = 1:nbl ; Qbl[k] = similar(Xbl[1], n, nlv) ; end
    Tb = list(nlv, Matrix{Float64})
    for a = 1:nlv ; Tb[a] = similar(Xbl[1], n, nbl) ; end
    Qb = list(nlv, Matrix{Float64})
    for a = 1:nlv ; Qb[a] = similar(Xbl[1], n, nbl) ; end
    lb = similar(Xbl[1], nbl, nlv)
    Wbl = list(nbl, Matrix{Float64})
    for k = 1:nbl ; Wbl[k] = similar(Xbl[1], p[k], nlv) ; end
    W = similar(Xbl[1], nbl, nlv)
    mu = similar(Xbl[1], nlv)
    niter = zeros(nlv)
    # End
    @inbounds for a = 1:nlv
        X = reduce(hcat, Xbl)
        u .= nipals(X).u    # u = "t" in Mang2019
        #u = X[:, 1] ; u ./= norm(u)
        cont = true
        iter = 1
        while cont
            u0 = copy(u)  
            for k = 1:nbl
                if tau == 0       
                    invCx = inv(Xbl[k]' * Xbl[k])
                elseif tau == 1   
                    invCx = Diagonal(ones(p[k]))
                else
                    I = Diagonal(ones(p[k]))
                    invCx = inv((1 - tau) * Xbl[k]' * Xbl[k] + tau * I)
                end
                wktild = invCx * (Xbl[k]' * u)
                dk = norm(wktild)
                mul!(qk, Xbl[k], wktild)    # qk = "tk" in Mang2019   
                Qb[a][:, k] .= qk
                Qbl[k][:, a] .= qk
                Tb[a][:, k] .= qk / dk    
                Tbl[k][:, a] .= (1 ./ sqrtw) .* Tb[a][:, k]
                Wbl[k][:, a] .= wktild / dk   
                lb[k, a] = dk^2             
            end
            if !wcov
                q .= rowsum(Qb[a])
                W[:, a] .= nipals(Tb[a]).v
            else
                q .= rowsum(lb[:, a]' .* Qb[a])
                W[:, a] .= nipals(Qb[a]).v
            end
            
            mu[a] = norm(q)   
            u .= q / mu[a]
            dif = sum((u .- u0).^2)
            iter = iter + 1
            if (dif < tol) || (iter > maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        U[:, a] .= u
        @inbounds for k = 1:nbl
            if deflat == "global"
                Xbl[k] .-= u * u' * Xbl[k]
            end
            if deflat == "can"
                #z = Qb[a][:, k]
                z = Tb[a][:, k]
                b = z' * Xbl[k] / dot(z, z)
                Xbl[k] .-= z * b
            end
        end
    end
    T = Diagonal(1 ./ sqrtw) *  (sqrt.(mu)' .* U)
    if !wcov
        Mbpca(T, U, W, Tbl, Tb, Wbl, lb, mu,
            bscales, xmeans, xscales, weights, niter)
    else 
        Comdim(T, U, W, Tbl, Tb, Wbl, lb, mu,
            bscales, xmeans, xscales, weights, niter)
    end
end









