"""
    comdim(; kwargs...)
    comdim(Xbl; kwargs...)
    comdim(Xbl, weights::Weight; kwargs...)
    comdim!(Xbl::Matrix, weights::Weight; kwargs...)
Common components and specific weights analysis (ComDim, *aka* CCSWA).
* `Xbl` : List of blocks (vector of matrices) of X-data. 
    Typically, output of function `mblock`.  
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
    * `bscal` : Type of block scaling. See function `blockscal`
        for possible values.
* `tol` : Tolerance value for convergence (Nipals).
* `maxit` : Maximum number of iterations (Nipals).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

"SVD" algorithm of Hannafi & Qannari 2008 p.84.

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
* `explvarx` : Proportion of the total inertia of X 
    (sum of the squared norms of the 
    blocks) explained by each global score.
* `explvarxx` : Proportion of the XX' total inertia 
    (sum of the squared norms of the products X_k * X_k') 
    explained by each global score (= indicator "V" in Qannari 
    et al. 2000, Hanafi et al. 2008).
* `sal2` : Proportion of the squared saliences
    of each block within each global score. 
* `contr_block` : Contribution of each block 
    to the global scores (= proportions of the saliences 
    "lambda" within each score).
    * `explX` : Proportion of the inertia of the blocks 
    explained by each global score.
* `corx2t` : Correlation between the global scores 
    and the original variables.  
* `cortb2t` : Correlation between the global scores 
    and the block scores.
* `rv` : RV coefficient. 
* `lg` : Lg coefficient. 

## References
Cariou, V., Qannari, E.M., Rutledge, D.N., Vigneau, E., 2018. 
ComDim: From multiblock data analysis to path modeling. Food 
Quality and Preference, Sensometrics 2016: Sensometrics-by-the-Sea 
67, 27–34. https://doi.org/10.1016/j.foodqual.2017.02.012

Cariou, V., Jouan-Rimbaud Bouveresse, D., Qannari, E.M., 
Rutledge, D.N., 2019. Chapter 7 - ComDim Methods for the Analysis 
of Multiblock Data in a Data Fusion Perspective, in: Cocchi, M. (Ed.), 
Data Handling in Science and Technology, 
Data Fusion Methodology and Applications. Elsevier, pp. 179–204. 
https://doi.org/10.1016/B978-0-444-63984-4.00007-7

Ghaziri, A.E., Cariou, V., Rutledge, D.N., Qannari, E.M., 2016. 
Analysis of multiblock datasets using ComDim: Overview and extension 
to the analysis of (K + 1) datasets. Journal of Chemometrics 30, 
420–429. https://doi.org/10.1002/cem.2810

Hanafi, M., 2008. Nouvelles propriétés de l’analyse en composantes 
communes et poids spécifiques. Journal de la société française 
de statistique 149, 75–97.

Qannari, E.M., Wakeling, I., Courcoux, P., MacFie, H.J.H., 2000. 
Defining the underlying sensory dimensions. Food Quality and 
Preference 11, 151–154. 
https://doi.org/10.1016/S0950-3293(99)00069-5

## Examples
```julia
using JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "ham.jld2") 
@load db dat
pnames(dat) 
X = dat.X
group = dat.group
listbl = [1:11, 12:19, 20:25]
Xbl = mblock(X[1:6, :], listbl)
Xblnew = mblock(X[7:8, :], listbl)
n = nro(Xbl[1]) 

nlv = 3
bscal = :frob
scal = false
#scal = true
mod = comdim(; nlv, bscal, scal)
fit!(mod, Xbl)
pnames(mod) 
pnames(mod.fm)
## Global scores 
@head mod.fm.T
@head transf(mod, Xbl)
transf(mod, Xblnew)
## Blocks scores
i = 1
@head mod.fm.Tbl[i]
@head transfbl(mod, Xbl)[i]

res = summary(mod, Xbl) ;
pnames(res) 
res.explvarx
res.explvarxx
res.sal2 
res.contr_block
res.explX   # = mod.fm.lb if bscal = :frob
rowsum(Matrix(res.explX))
res.corx2t 
res.cortb2t
res.rv
```
"""
function comdim(Xbl; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    weights = mweight(ones(Q, n))
    comdim(Xbl, weights; kwargs...)
end

function comdim(Xbl, weights::Weight; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)  
    zXbl = list(Matrix{Q}, nbl)
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    comdim!(zXbl, weights; kwargs...)
end

function comdim!(Xbl::Vector, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs) 
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)
    n = nro(Xbl[1])
    nlv = par.nlv
    sqrtw = sqrt.(weights.w)
    fmsc = blockscal(Xbl, weights; bscal = par.bscal,  
        centr = true, scal = par.scal)
    transf!(fmsc, Xbl)
    # Row metric
    @inbounds for k = 1:nbl
        Xbl[k] .= sqrtw .* Xbl[k]
    end
    ## Pre-allocation
    u = similar(Xbl[1], n)
    U = similar(Xbl[1], n, nlv)
    tk = copy(u)
    Tbl = list(Matrix{Q}, nbl)
    for k = 1:nbl ; Tbl[k] = similar(Xbl[1], n, nlv) ; end
    Tb = list(Matrix{Q}, nlv)
    for a = 1:nlv ; Tb[a] = similar(Xbl[1], n, nbl) ; end
    Wbl = list(Matrix{Q}, nbl)
    for k = 1:nbl ; Wbl[k] = similar(Xbl[1], nco(Xbl[k]), nlv) ; end
    lb = similar(Xbl[1], nbl, nlv)
    mu = similar(Xbl[1], nlv)
    TB = similar(Xbl[1], n, nbl)
    W = similar(Xbl[1], nbl, nlv)
    niter = zeros(nlv)
    # End
    res = 0
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
            if (dif < par.tol) || (iter > par.maxit)
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
        fmsc, weights, niter, kwargs, par)
end

""" 
    transf(object::Comdim, Xbl; nlv = nothing)
    transfbl(object::Comdim, Xbl; nlv = nothing)
Compute latent variables (LVs = scores T) from 
    a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transf(object::Comdim, Xbl; nlv = nothing)
    transf_all(object, Xbl; nlv).T
end

function transfbl(object::Comdim, Xbl; nlv = nothing)
    transf_all(object, Xbl; nlv).Tbl
end

function transf_all(object::Comdim, Xbl; nlv = nothing)
    Q = eltype(Xbl[1][1, 1])
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    nbl = length(Xbl)
    m = size(Xbl[1], 1)
    zXbl = transf(object.fmsc, Xbl)    
    U = similar(zXbl[1], m, nlv)
    TB = similar(zXbl[1], m, nbl)
    Tbl = list(Matrix{Q}, nbl)
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
    T = sqrt.(object.mu[1:nlv])' .* U
    (T = T, Tbl)
end

"""
    summary(object::Comdim, Xbl)
Summarize the fitted model.
* `object` : The fitted model.
* `Xbl` : The X-data that was used to 
    fit the model.
""" 
function Base.summary(object::Comdim, Xbl)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)
    nlv = nco(object.T)
    sqrtw = sqrt.(object.weights.w)
    zXbl = transf(object.fmsc, Xbl)
    @inbounds for k = 1:nbl
        zXbl[k] .= sqrtw .* zXbl[k]
    end
    ## Explained_X by global scores
    sstot = zeros(Q, nbl)
    @inbounds for k = 1:nbl
        sstot[k] = ssq(zXbl[k])
    end
    tt = colsum(object.lb)    
    pvar = tt / sum(sstot)
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(lv = 1:nlv, var = tt, 
        pvar = pvar, cumpvar = cumpvar)
    ## Explained_XXt (indicator "V")
    S = list(Matrix{Q}, nbl)
    sstot_xx = 0 
    @inbounds for k = 1:nbl
        S[k] = zXbl[k] * zXbl[k]'
        sstot_xx += ssq(S[k])
    end
    tt = object.mu
    pvar = tt / sstot_xx
    cumpvar = cumsum(pvar)
    explvarxx = DataFrame(lv = 1:nlv, var = tt, 
        pvar = pvar, cumpvar = cumpvar)
    ## Proportions of squared saliences
    sal2 = copy(object.lb)
    for a = 1:nlv
        sal2[:, a] .= object.lb[:, a].^2 / object.mu[a]
    end
    sal2 = DataFrame(sal2, string.("lv", 1:nlv))
    ## Contribution of the blocks to global 
    ## scores = lb proportions (contrib)
    z = fscale(object.lb, colsum(object.lb))
    contr_block = DataFrame(z, string.("lv", 1:nlv))
    ## Proportion of inertia explained for 
    ## each block (explained.X)
    # = object.lb if bscal = :frob 
    z = fscale((object.lb)', sstot)'
    explX = DataFrame(z, string.("lv", 1:nlv))
    ## Correlation between the original variables and 
    ## the global scores (globalcor)
    X = reduce(hcat, zXbl)
    z = cor(X, object.U)  
    corx2t = DataFrame(z, string.("lv", 1:nlv))  
    ## Correlation between the block scores and 
    ## the global scores (cor.g.b)
    z = list(Matrix{Q}, nlv)
    @inbounds for a = 1:nlv
        z[a] = cor(object.Tb[a], object.U[:, a])
    end
    cortb2t = DataFrame(reduce(hcat, z), 
        string.("lv", 1:nlv))
    ## RV 
    X = vcat(zXbl, [sqrtw .* object.T])
    nam = [string.("block", 1:nbl) ; "T"]
    res = rv(X)
    zrv = DataFrame(res, nam)
    ## Lg
    res = lg(X)
    zlg = DataFrame(res, nam)
    (explvarx = explvarx, explvarxx, sal2, contr_block, 
        explX, corx2t, cortb2t, rv = zrv, lg = zlg)
end


