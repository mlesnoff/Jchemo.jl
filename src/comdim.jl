struct Comdim
    T::Array{Float64} 
    U::Array{Float64}
    W::Array{Float64}
    Tbl::Vector{Array{Float64}}
    Tb::Vector{Array{Float64}}
    Wbl::Vector{Array{Float64}}
    lb::Array{Float64}
    mu::Vector{Float64}
    bscales::Vector{Float64}
    xmeans::Vector{Vector{Float64}}
    xscales::Vector{Vector{Float64}}
    weights::Vector{Float64}
    niter::Vector{Float64}
end

"""
    comdim(Xbl, weights = ones(nro(Xbl[1])); nlv,
        bscal = "none", tol = sqrt(eps(1.)), maxit = 200,
        scal = false)
    comdim!(Xbl, weights = ones(nro(Xbl[1])); nlv,
        bscal = "none", tol = sqrt(eps(1.)), maxit = 200,
        scal = false)
Common components and specific weights analysis (ComDim = CCSWA).
* `Xbl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling (`"none"`, `"frob"`). 
    See functions `blockscal`.
* `tol` : Tolerance value for convergence.
* `niter` : Maximum number of iterations.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

This version corresponds to the "SVD" algorithm of Hannafi & Qannari 2008 p.84.

The function returns several objects, in particular:
* `T` : The non normed global scores.
* `U` : The normed global scores.
* `W` : The global loadings.
* `Tbl` : The block scores (grouped by blocks, in the original scale).
* `Tb` : The block scores (grouped by LV, in the metric scale).
* `Wbl` : The block loadings.
* `lb` : The specific weights (saliences) "lambda".
* `mu` : The sum of the squared saliences.

Function `summary` returns: 
* `explvarx` : Proportion of the X total inertia (sum of the squared norms of the 
    blocks) explained by each global score.
* `explvarxx` : Proportion of the XX' total inertia (sum of the squared norms of the
    products X_k * X_k') explained by each global score 
    (= indicator "V" in Qannari et al. 2000, Hanafi et al. 2008).
* `sal2` : Proportion of the squared saliences (specific weights)
    of each block within each global score. 
* `contr_block` : Contribution of each block to the global scores 
    (= proportions of the saliences "lambda" within each score)
* `explX` : Proportion of the inertia of the blocks explained by each global score.
* `cort2x` : Correlation between the global scores and the original variables.  
* `cort2tb` : Correlation between the global scores and the block scores.
* `rv` : RV coefficient. 
* `lg` : Lg coefficient. 

## References
Cariou, V., Qannari, E.M., Rutledge, D.N., Vigneau, E., 2018. ComDim: From multiblock data 
analysis to path modeling. Food Quality and Preference, Sensometrics 2016: 
Sensometrics-by-the-Sea 67, 27–34. https://doi.org/10.1016/j.foodqual.2017.02.012

Cariou, V., Jouan-Rimbaud Bouveresse, D., Qannari, E.M., Rutledge, D.N., 2019. 
Chapter 7 - ComDim Methods for the Analysis of Multiblock Data in a Data Fusion 
Perspective, in: Cocchi, M. (Ed.), Data Handling in Science and Technology, 
Data Fusion Methodology and Applications. Elsevier, pp. 179–204. 
https://doi.org/10.1016/B978-0-444-63984-4.00007-7

Ghaziri, A.E., Cariou, V., Rutledge, D.N., Qannari, E.M., 2016. Analysis of multiblock 
datasets using ComDim: Overview and extension to the analysis of (K + 1) datasets. 
Journal of Chemometrics 30, 420–429. https://doi.org/10.1002/cem.2810

Hanafi, M., 2008. Nouvelles propriétés de l’analyse en composantes communes et 
poids spécifiques. Journal de la société française de statistique 149, 75–97.

Qannari, E.M., Wakeling, I., Courcoux, P., MacFie, H.J.H., 2000. Defining the underlying 
sensory dimensions. Food Quality and Preference 11, 151–154. 
https://doi.org/10.1016/S0950-3293(99)00069-5

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "ham.jld2") 
@load db dat
pnames(dat) 

X = dat.X
group = dat.group
listbl = [1:11, 12:19, 20:25]
Xbl = mblock(X, listbl)
# "New" = first two rows of Xbl 
Xbl_new = mblock(X[1:2, :], listbl)

bscal = "none"
#bscal = "frob"
fm = comdim(Xbl; nlv = 4, bscal = bscal) ;
fm.U
fm.T
Jchemo.transform(fm, Xbl)
Jchemo.transform(fm, Xbl_new) 

res = summary(fm, Xbl) ;
fm.lb
rowsum(fm.lb)
fm.mu
res.explvarx
res.explvarxx
res.explX # = fm.lb if bscal = "frob"
rowsum(Matrix(res.explX))
res.contr_block
res.sal2
colsum(Matrix(res.sal2))
res.cort2x 
res.cort2tb
res.rv
```
"""
function comdim(Xbl, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "none", tol = sqrt(eps(1.)), maxit = 200,
        scal = false)
    nbl = length(Xbl)  
    zXbl = list(nbl, Matrix{Float64})
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    comdim!(zXbl, weights; nlv = nlv, 
        bscal = bscal, tol = tol, maxit = maxit, scal = scal)
end

## Approach Hannafi & Qannari 2008 p.84: "SVD" algorithm
## Normed global score u = 1st left singular vector of SVD of TB,
## where TB concatenates the weighted block-scores 
function comdim!(Xbl, weights = ones(nro(Xbl[1])); nlv,
        bscal = "none", tol = sqrt(eps(1.)), maxit = 200,
        scal = false)
    nbl = length(Xbl)
    n = nro(Xbl[1])
    weights = mweight(weights)
    sqrtw = sqrt.(weights)
    sqrtD = Diagonal(sqrtw)
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
    @inbounds for k = 1:nbl
        Xbl[k] .= sqrtw .* Xbl[k]
    end
    # Pre-allocation
    u = similar(Xbl[1], n)
    U = similar(Xbl[1], n, nlv)
    tk = copy(u)
    Tbl = list(nbl, Matrix{Float64})
    for k = 1:nbl ; Tbl[k] = similar(Xbl[1], n, nlv) ; end
    Tb = list(nlv, Matrix{Float64})
    for a = 1:nlv ; Tb[a] = similar(Xbl[1], n, nbl) ; end
    Wbl = list(nbl, Matrix{Float64})
    for k = 1:nbl ; Wbl[k] = similar(Xbl[1], p[k], nlv) ; end
    lb = similar(Xbl[1], nbl, nlv)
    mu = similar(Xbl[1], nlv)
    TB = similar(Xbl[1], n, nbl)
    W = similar(Xbl[1], nbl, nlv)
    niter = zeros(nlv)
    # End
    @inbounds for a = 1:nlv
        X = reduce(hcat, Xbl)
        u .= nipals(X).u
        iter = 1
        cont = true
        while cont
            u0 = copy(u)
            for k = 1:nbl
                wk = Xbl[k]' * u      # = wktild
                dk = norm(wk)         # = alphak = abs.(dot(tk, u))
                wk ./= dk             # = wk (= normed)
                mul!(tk, Xbl[k], wk) 
                Tb[a][:, k] .= tk
                Tbl[k][:, a] .= (1 ./ sqrtw) .* tk
                TB[:, k] = dk * tk    # = Qb (qk = dk * tk)
                Wbl[k][:, a] .= wk
                lb[k, a] = dk^2
            end
            res = nipals(TB)
            u .= res.u
            dif = sum((u - u0).^2)
            iter = iter + 1
            if (dif < tol) || (iter > maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        U[:, a] .= u
        W[:, a] .= res.v
        mu[a] = res.sv^2   # = sum(lb.^2)   
        @inbounds for k = 1:nbl
            Xbl[k] .-= u * (u' * Xbl[k])
            # Same as:
            #Px = sqrt(lb[k, a]) * Wbl[k][:, a]'
            #Xbl[k] .-= u * Px
        end
    end
    T = Diagonal(1 ./ sqrtw) * (sqrt.(mu)' .* U)
    Comdim(T, U, W, Tbl, Tb, Wbl, lb, mu, 
        bscales, xmeans, xscales, weights, niter)
end

""" 
    transform(object::Comdim, Xbl; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and X-data.
* `object` : The fitted model.
* `Xbl` : A list (vector) of blocks (matrices) of X-data for which LVs are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transform(object::Comdim, Xbl; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    nbl = length(Xbl)
    m = size(Xbl[1], 1)
    zXbl = list(nbl, Matrix{Float64})
    Threads.@threads for k = 1:nbl
        zXbl[k] = cscale(Xbl[k], object.xmeans[k], object.xscales[k])
    end
    zXbl = blockscal(zXbl, object.bscales).X
    U = similar(zXbl[1], m, nlv)
    TB = similar(zXbl[1], m, nbl)
    Tbl = list(nbl, Matrix{Float64})
    for k = 1:nbl ; Tbl[k] = similar(zXbl[1], m, nlv) ; end
    u = similar(zXbl[1], m)
    tk = copy(u)
    for a = 1:nlv
        for k = 1:nbl
            tk .= zXbl[k] * object.Wbl[k][:, a]
            TB[:, k] .= tk
            Tbl[k][:, a] .= tk
        end
        TB .= sqrt.(object.lb[:, a])' .* TB
        u .= 1 / sqrt(object.mu[a]) * TB * object.W[:, a]
        U[:, a] .= u
        @inbounds for k = 1:nbl
            Px = sqrt(object.lb[k, a]) * object.Wbl[k][:, a]'
            zXbl[k] .-= u * Px
        end
    end
    T = sqrt.(object.mu)' .* U
    (T = T, Tbl)
end

"""
    summary(object::Comdim, Xbl)
Summarize the fitted model.
* `object` : The fitted model.
* `Xbl` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Comdim, Xbl)
    nbl = length(Xbl)
    nlv = size(object.T, 2)
    sqrtw = sqrt.(object.weights)
    zXbl = list(nbl, Matrix{Float64})
    Threads.@threads for k = 1:nbl
        zXbl[k] = cscale(Xbl[k], object.xmeans[k], object.xscales[k])
    end
    zXbl = blockscal(zXbl, object.bscales).X
    @inbounds for k = 1:nbl
        zXbl[k] .= sqrtw .* zXbl[k]
    end
    # Explained_X
    sstot = zeros(nbl)
    @inbounds for k = 1:nbl
        sstot[k] = ssq(zXbl[k])
    end
    tt = colsum(object.lb)    
    pvar = tt / sum(sstot)
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(lv = 1:nlv, var = tt, pvar = pvar, 
        cumpvar = cumpvar)
    # Explained_XXt (indicator "V")
    S = list(nbl, Matrix{Float64})
    sstot_xx = 0 
    @inbounds for k = 1:nbl
        S[k] = zXbl[k] * zXbl[k]'
        sstot_xx += ssq(S[k])
    end
    tt = object.mu
    pvar = tt / sstot_xx
    cumpvar = cumsum(pvar)
    explvarxx = DataFrame(lv = 1:nlv, var = tt, pvar = pvar, 
        cumpvar = cumpvar)
    # Proportions of squared saliences
    sal2 = copy(object.lb)
    for a = 1:nlv
        sal2[:, a] .= object.lb[:, a].^2 / object.mu[a]
    end
    sal2 = DataFrame(sal2, string.("pc", 1:nlv))
    # Contribution of the blocks to global scores = lb proportions (contrib)
    z = scale(object.lb, colsum(object.lb))
    contr_block = DataFrame(z, string.("pc", 1:nlv))
    # Proportion of inertia explained for each block (explained.X)
    # = object.lb if bscal = "frob" 
    z = scale((object.lb)', sstot)'
    explX = DataFrame(z, string.("pc", 1:nlv))
    # Correlation between the global scores and the original variables (globalcor)
    X = reduce(hcat, zXbl)
    z = cor(X, object.U)  
    cort2x = DataFrame(z, string.("pc", 1:nlv))  
    # Correlation between the global scores and the block_scores (cor.g.b)
    z = list(nlv, Matrix{Float64})
    @inbounds for a = 1:nlv
        z[a] = cor(object.Tb[a], object.U[:, a])
    end
    cort2tb = DataFrame(reduce(hcat, z), string.("pc", 1:nlv))
    # RV 
    X = vcat(zXbl, [object.T])
    nam = [string.("block", 1:nbl) ; "T"]
    res = rv(X)
    zrv = DataFrame(res, nam)
    # Lg
    res = lg(X)
    zlg = DataFrame(res, nam)
    (explvarx = explvarx, explvarxx, sal2, contr_block, explX, 
        cort2x, cort2tb, rv = zrv, lg = zlg)
end


