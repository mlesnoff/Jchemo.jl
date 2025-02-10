"""
    comdim(; kwargs...)
    comdim(Xbl; kwargs...)
    comdim(Xbl, weights::Weight; kwargs...)
    comdim!(Xbl::Matrix, weights::Weight; kwargs...)
Common components and specific weights analysis (CCSWA, a.k.a ComDim).
* `Xbl` : List of blocks (vector of matrices) of X-data. 
    Typically, output of function `mblock`.  
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling. See function `blockscal` for possible values.
* `tol` : Tolerance value for convergence (Nipals).
* `maxit` : Maximum number of iterations (Nipals).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

"SVD" algorithm of Hannafi & Qannari 2008 p.84.

The function returns several objects, in particular:
* `T` : The global scores (not-normed).
* `U` : The normed global scores.
* `W` : The normed block weights.
* `Tb` : The block scores (in the metric scale), returned **grouped by LV**.
* `Tbl` : The block scores (in the original scale), returned **grouped by block**.
* `Vbl` : The normed block loadings.
* `lb` : The block specific weights (saliences) 'lambda'.
* `mu` : The sum of the squared saliences per LV.

Function `summary` returns: 

* `explvarx` : Proportion of the total inertia of X (squared Frobenious norm) 
    explained by each global score.
* `explvarxx` : Proportion of the XX' total inertia explained by each global 
    score (= indicator "V" in Qannari et al. 2000, Hanafi et al. 2008).
* `explX` : Proportion of the inertia of each block explained by each global score.
* `psal2` : Proportion of the squared saliences of each block within each global score. 
* `contr_block` : Contribution of each block to the global scores. 
* `cortbl2t` : Correlation between the block scores and the global scores.
* `corx2t` : Correlation between the original variables and the global scores.  
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
using Jchemo, JchemoData, JLD2
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
model = comdim(; nlv, bscal, scal)
fit!(model, Xbl)
pnames(model) 
pnames(model.fitm)
## Global scores 
@head model.fitm.T
@head transf(model, Xbl)
transf(model, Xblnew)
## Blocks scores
i = 1
@head model.fitm.Tbl[i]
@head transfbl(model, Xbl)[i]

res = summary(model, Xbl) ;
pnames(res) 
res.explvarx
res.explvarxx
res.psal2 
res.contr_block
res.explX   # = model.fitm.lb if bscal = :frob
rowsum(Matrix(res.explX))
res.cortbl2t
res.corx2t 
res.rv
```
"""
comdim(; kwargs...) = JchemoModel(comdim, nothing, kwargs)

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
    @inbounds for k in eachindex(Xbl)
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    comdim!(zXbl, weights; kwargs...)
end

function comdim!(Xbl::Vector, weights::Weight; kwargs...)
    par = recovkw(ParMbpca, kwargs).par 
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)
    n = nro(Xbl[1])
    nlv = par.nlv
    fitmbl = blockscal(Xbl, weights; centr = true, scal = par.scal, bscal = par.bscal)
    transf!(fitmbl, Xbl)
    # Row metric
    sqrtw = sqrt.(weights.w)
    invsqrtw = 1 ./ sqrtw
    @inbounds for k in eachindex(Xbl)
        Xbl[k] .= sqrtw .* Xbl[k]
    end
    ## Pre-allocation
    u = similar(Xbl[1], n)
    U = similar(Xbl[1], n, nlv)
    tk = copy(u)
    Tbl = list(Matrix{Q}, nbl)
    for k in eachindex(Xbl) ; Tbl[k] = similar(Xbl[1], n, nlv) ; end
    Tb = list(Matrix{Q}, nlv)
    for a = 1:nlv ; Tb[a] = similar(Xbl[1], n, nbl) ; end
    Vbl = list(Matrix{Q}, nbl)
    for k in eachindex(Xbl) ; Vbl[k] = similar(Xbl[1], nco(Xbl[k]), nlv) ; end
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
            for k in eachindex(Xbl)
                vk = Xbl[k]' * u      # = wktild
                dk = normv(vk)        # = alphak = abs.(dot(tk, u))
                vk ./= dk             # = vk (= normed)
                mul!(tk, Xbl[k], vk) 
                Tb[a][:, k] .= tk
                Tbl[k][:, a] .= (1 ./ sqrtw) .* tk
                TB[:, k] = dk * tk    # = Qb (qk = dk * tk)
                Vbl[k][:, a] .= vk
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
        U[:, a] .= u .* invsqrtw
        W[:, a] .= res.v
        mu[a] = res.sv^2   # = sum(lb.^2)   
        @inbounds for k in eachindex(Xbl)
            Xbl[k] .-= u * (u' * Xbl[k])
            # Same as:
            #Vx = sqrt(lb[k, a]) * Vbl[k][:, a]'
            #Xbl[k] .-= u * Vx
        end
    end
    T = sqrt.(mu)' .* U    
    Comdim(T, U, W, Tb, Tbl, Vbl, lb, mu, fitmbl, weights, niter, par)
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
    zXbl = transf(object.fitmbl, Xbl)    
    U = similar(zXbl[1], m, nlv)
    TB = similar(zXbl[1], m, nbl)
    Tbl = list(Matrix{Q}, nbl)
    for k in eachindex(Xbl) ; Tbl[k] = similar(zXbl[1], m, nlv) ; end
    u = similar(zXbl[1], m)
    tk = copy(u)
    for a = 1:nlv
        for k in eachindex(Xbl)
            tk .= zXbl[k] * object.Vbl[k][:, a]
            TB[:, k] .= tk
            Tbl[k][:, a] .= tk
        end
        TB .= sqrt.(object.lb[:, a])' .* TB
        u .= 1 / sqrt(object.mu[a]) * TB * object.W[:, a]
        U[:, a] .= u
        @inbounds for k in eachindex(Xbl)
            Vx = sqrt(object.lb[k, a]) * object.Vbl[k][:, a]'
            zXbl[k] .-= u * Vx
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
    zXbl = transf(object.fitmbl, Xbl)
    sqrtw = sqrt.(object.weights.w)
    @inbounds for k in eachindex(Xbl)
        fweight!(zXbl[k], sqrtw)
    end
    X = fconcat(zXbl)
    ## Proportion of the X-inertia explained per global LV
    sstot = zeros(Q, nbl)
    @inbounds for k in eachindex(Xbl)
        sstot[k] = frob2(zXbl[k])
    end
    tt = colsum(object.lb)    
    #tt = colnorm(object.T, object.weights).^2 
    pvar = tt / sum(sstot)
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(lv = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    ## Explained XXt (indicator 'V') per global LV
    S = list(Matrix{Q}, nbl)
    sstot_xx = 0 
    @inbounds for k in eachindex(Xbl)
        S[k] = zXbl[k] * zXbl[k]'
        sstot_xx += frob2(S[k])
    end
    tt = object.mu
    pvar = tt / sstot_xx
    cumpvar = cumsum(pvar)
    explvarxx = DataFrame(lv = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    ## Within each block, proportion of the block-inertia explained by each global LV
    ## = object.lb if bscal = :frob 
    z = fscale(object.lb', sstot)'
    nam = string.("lv", 1:nlv)
    explX = DataFrame(z, nam)
    ## Poportion of squared saliences
    psal2 = copy(object.lb)
    @inbounds for a = 1:nlv
        psal2[:, a] .= object.lb[:, a].^2 / object.mu[a]
    end
    psal2 = DataFrame(psal2, nam)
    ## Contribution of the blocks to global LVs
    # = lb proportions
    z = fscale(object.lb, colsum(object.lb))
    contr_block = DataFrame(z, nam)
    ## Correlation between the block LVs and the global LVs
    z = zeros(Q, nbl, nlv)
    for k in eachindex(Xbl), a = 1:nlv 
        z[k, a] = corv(object.Tbl[k][:, a], object.T[:, a], object.weights) 
    end
    cortbl2t = DataFrame(z, nam)
    ## Correlation between the original variables and the global LVs 
    z = cor(X, object.U)  
    corx2t = DataFrame(z, nam)  
    ## RV 
    nam = [string.("block", 1:nbl) ; "T"]
    X = vcat(zXbl, [fweight(object.T, sqrtw)])
    res = rv(X)
    zrv = DataFrame(res, nam)
    ## Lg
    res = lg(X)
    zlg = DataFrame(res, nam)
    (explvarx = explvarx, explvarxx, explX, psal2, contr_block, cortbl2t, corx2t, 
        rv = zrv, lg = zlg)
end


