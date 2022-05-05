struct MbpcaComdim
    T::Array{Float64} 
    U::Array{Float64}
    W::Array{Float64}
    Tb::Vector{Array{Float64}}
    W_bl::Vector{Array{Float64}}
    lb::Array{Float64}
    mu::Vector{Float64}
    xmeans::Vector{Vector{Float64}}
    scal::Vector{Float64}
    weights::Vector{Float64}
    niter::Vector{Float64}
end

"""
    mbpca_comdim_s(X_bl, weights = ones(size(X_bl[1], 1)); nlv,
        bscal = "none", tol = sqrt(eps(1.)), maxit = 200)
Common components and specific weights analysis (CCSWA, ComDim).
* `X_bl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `weights` : Weights of the observations (rows). 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscal` : Type of block scaling (`"none"`, `"frob"`). 
    See functions `blockscal`.
* `tol` : Tolerance value for convergence.
* `niter` : Maximum number of iterations.

This version corresponds to the "SVD" algorithm of Hannafi & Qannari 2008 p.84.

`weights` is internally normalized to sum to 1.

The function returns several objects, in particular:
* `T` : The non normed global scores.
* `U` : The normed global scores.
* `W` : The global loadings.
* `Tb` : The block scores.
* `W_bl` : The block loadings.
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
mypath = dirname(dirname(pathof(Jchemo)))
db = joinpath(mypath, "data", "ham.jld2") 
@load db dat
pnames(dat) 

X = dat.X
group = dat.group
listbl = [1:11, 12:19, 20:25]
X_bl = mblock(X, listbl)
# "New" = first two rows of X_bl 
X_bl_new = mblock(X[1:2, :], listbl)

bscal = "none"
#bscal = "frob"
fm = mbpca_comdim_s(X_bl; nlv = 4, bscal = bscal) ;
fm.U
fm.T
Jchemo.transform(fm, X_bl)
Jchemo.transform(fm, X_bl_new) 

res = Jchemo.summary(fm, X_bl) ;
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
function mbpca_comdim_s(X_bl, weights = ones(size(X_bl[1], 1)); nlv,
        bscal = "none", tol = sqrt(eps(1.)), maxit = 200)
    nbl = length(X_bl)
    X = copy(X_bl)
    n = size(X[1], 1)
    weights = mweight(weights)
    sqrtw = sqrt.(weights)
    sqrtD = Diagonal(sqrtw)
    xmeans = list(nbl, Vector{Float64})
    p = fill(0, nbl)
    @inbounds for k = 1:nbl
        p[k] = size(X[k], 2)
        xmeans[k] = colmean(X[k], weights)   
        X[k] = center(X[k], xmeans[k])
    end
    bscal == "none" ? scal = ones(nbl) : nothing
    if bscal == "frob"
        res = blockscal_frob(X, weights) 
        scal = res.scal
        X = res.X
    end
    @inbounds for k = 1:nbl
        X[k] .= sqrtD * X[k]
    end
    # Pre-allocation
    u = similar(X[1], n)
    U = similar(X[1], n, nlv)
    tb = copy(u)
    Tb = list(nlv, Matrix{Float64})
    for a = 1:nlv ; Tb[a] = similar(X[1], n, nbl) ; end
    W_bl = list(nbl, Matrix{Float64})
    for k = 1:nbl ; W_bl[k] = similar(X[1], p[k], nlv) ; end
    lb = similar(X[1], nbl, nlv)
    mu = similar(X[1], nlv)
    TB = similar(X[1], n, nbl)
    W = similar(X[1], nbl, nlv)
    niter = zeros(nlv)
    # End
    @inbounds for a = 1:nlv
        zX = reduce(hcat, X)
        u .= nipals(zX).u
        iter = 1
        cont = true
        while cont
            u0 = copy(u)
            for k = 1:nbl
                wb = X[k]' * u 
                wb ./= norm(wb)
                mul!(tb, X[k], wb) 
                alpha = abs.(dot(tb, u))
                TB[:, k] = alpha * tb
                lb[k, a] = alpha^2
                Tb[a][:, k] .= tb
                W_bl[k][:, a] .= wb
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
            X[k] .-= u * (u' * X[k])
            # Same as:
            #Px = sqrt(lb[k, a]) * W_bl[k][:, a]'
            #X[k] .-= u * Px
        end
    end
    T = Diagonal(1 ./ sqrtw) * (sqrt.(mu)' .* U)
    MbpcaComdim(T, U, W, Tb, W_bl, lb, mu, 
        xmeans, scal, weights, niter)
end

""" 
    transform(object::MbpcaComdim, X_bl; nlv = nothing)
Compute components (scores matrix "T") from a fitted model and X-data.
* `object` : The maximal fitted model.
* `X_bl` : A list (vector) of blocks (matrices) of X-data for which LVs are computed.
* `nlv` : Nb. components to compute. If nothing, it is the maximum nb. PCs.
""" 
function transform(object::MbpcaComdim, X_bl; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    nbl = length(X_bl)
    X = copy(X_bl)
    m = size(X[1], 1)
    @inbounds for k = 1:nbl
        X[k] = center(X[k], object.xmeans[k])
    end
    X = blockscal(X; scal = object.scal).X
    U = similar(X[1], m, nlv)
    TB = similar(X[1], m, nbl)
    u = similar(X[1], m)
    for a = 1:nlv
        for k = 1:nbl
            TB[:, k] .= X[k] * object.W_bl[k][:, a]
        end
        TB .= sqrt.(object.lb[:, a])' .* TB
        u .= 1 / sqrt(object.mu[a]) * TB * object.W[:, a]
        U[:, a] .= u
        @inbounds for k = 1:nbl
            Px = sqrt(object.lb[k, a]) * object.W_bl[k][:, a]'
            X[k] .-= u * Px
        end
    end
    sqrt.(object.mu)' .* U # = T
end

"""
    summary(object::MbpcaComdim, X_bl)
Summarize the fitted model.
* `object` : The fitted model.
* `X_bl` : The X-data that was used to fit the model.
""" 
function summary(object::MbpcaComdim, X_bl)
    nbl = length(X_bl)
    nlv = size(object.T, 2)
    X = copy(X_bl)
    sqrtw = sqrt.(object.weights)
    sqrtD = Diagonal(sqrtw)
    @inbounds for k = 1:nbl
        X[k] = center(X[k], object.xmeans[k])
    end
    X = blockscal(X; scal = object.scal).X
    @inbounds for k = 1:nbl
        X[k] .= sqrtD * X[k]
    end
    # Explained_X
    sstot = zeros(nbl)
    @inbounds for k = 1:nbl
        sstot[k] = ssq(X[k])
    end
    tt = colsum(object.lb)    
    pvar = tt / sum(sstot)
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(pc = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    # Explained_XXt (indicator "V")
    S = list(nbl, Matrix{Float64})
    sstot_xx = 0 
    @inbounds for k = 1:nbl
        S[k] = X[k] * X[k]'
        sstot_xx += ssq(S[k])
    end
    tt = object.mu
    pvar = tt / sstot_xx
    cumpvar = cumsum(pvar)
    explvarxx = DataFrame(pc = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    # Prop saliences^2
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
    zX = reduce(hcat, X)
    z = cor(zX, object.U)  
    cort2x = DataFrame(z, string.("pc", 1:nlv))  
    # Correlation between the global scores and the block_scores (cor.g.b)
    z = list(nlv, Matrix{Float64})
    @inbounds for a = 1:nlv
        z[a] = cor(object.Tb[a], object.U[:, a])
    end
    cort2tb = DataFrame(reduce(hcat, z), string.("pc", 1:nlv))
    # RV 
    zX = vcat(X, [object.T])
    nam = [string.("block", 1:nbl) ; "T"]
    res = rv(zX)
    zrv = DataFrame(res, nam)
    # Lg
    res = lg(zX)
    zlg = DataFrame(res, nam)
    (explvarx = explvarx, explvarxx, sal2, contr_block, explX, 
        cort2x, cort2tb, rv = zrv, lg = zlg)
end


