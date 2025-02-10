"""
    mbpca(; kwargs...)
    mbpca(Xbl; kwargs...)
    mbpca(Xbl, weights::Weight; kwargs...)
    mbpca!(Xbl::Matrix, weights::Weight; kwargs...)
Consensus principal components analysis (CPCA, a.k.a MBPCA).
* `Xbl` : List of blocks (vector of matrices) of X-data. 
    Typically, output of function `mblock`.  
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling. See function `blockscal` for possible values.
* `tol` : Tolerance value for Nipals convergence.
* `maxit` : Maximum number of iterations (Nipals).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    is scaled by its uncorrected standard deviation 
    (before the block scaling).

CPCA algorithm (Westerhuis et a; 1998), a.k.a MBPCA, and reffered to as CPCA-W in 
Smilde et al. 2003. 

Apart eventual block scaling, the MBPCA is equivalent to the PCA of the 
horizontally concatenated matrix X = [X1 X2 ... Xk] (SUM-PCA in Smilde et al 2003).

The function returns several objects, in particular:
* `T` : The global scores (not-normed).
* `U` : The normed global scores.
* `W` : The normed block weights.
* `Tb` : The block scores (in the metric scale), returned **grouped by LV**.
* `Tbl` : The block scores (in the original scale), returned **grouped by block**.
* `Vbl` : The normed block loadings.
* `lb` : The block specific weights "lambda".
* `mu` : The sum of the specific weights (= eigen values of the global PCA).

Function `summary` returns: 
* `explvarx` : Proportion of the total inertia of X (squared Frobenious norm) 
    explained by each global score.
* `explX` : Proportion of the inertia of each block explained by each global score.
* `contr_block` : Contribution of each block to the global scores. 
* `cortbl2t` : Correlation between the block scores and the global scores.
* `corx2t` : Correlation between the original variables and the global scores.  
* `rv` : RV coefficient. 
* `lg` : Lg coefficient. 

## References
Mangamana, E.T., Cariou, V., Vigneau, E., Glèlè Kakaï, R.L., 
Qannari, E.M., 2019. Unsupervised multiblock data 
analysis: A unified approach and extensions. Chemometrics and 
Intelligent Laboratory Systems 194, 103856. 
https://doi.org/10.1016/j.chemolab.2019.103856

Smilde, A.K., Westerhuis, J.A., de Jong, S., 2003. A framework for sequential 
multiblock component methods. Journal of Chemometrics 17, 323–337.
 https://doi.org/10.1002/cem.811

Westerhuis, J.A., Kourti, T., MacGregor, J.F., 1998. Analysis 
of multiblock and hierarchical PCA and PLS models. Journal 
of Chemometrics 12, 301–321. 
https://doi.org/10.1002/(SICI)1099-128X(199809/10)12:5<301::AID-CEM515>3.0.CO;2-S

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
model = mbpca(; nlv, bscal, scal)
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
res.explX   # = model.fitm.lb if bscal = :frob
rowsum(Matrix(res.explX))
res.contr_block
res.cortbl2t
res.corx2t 
res.rv
```
"""
mbpca(; kwargs...) = JchemoModel(mbpca, nothing, kwargs)

function mbpca(Xbl; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    mbpca(Xbl, mweight(ones(Q, n)); kwargs...)
end

function mbpca(Xbl, weights::Weight; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)  
    zXbl = list(Matrix{Q}, nbl)
    @inbounds for k in eachindex(Xbl)
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    mbpca!(zXbl, weights; kwargs...)
end

function mbpca!(Xbl::Vector, weights::Weight; kwargs...)
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
        fweight!(Xbl[k], sqrtw)
    end
    ## Pre-allocation
    U = similar(Xbl[1], n, nlv)
    W = similar(Xbl[1], nbl, nlv)
    Tbl = list(Matrix{Q}, nbl)
    for k in eachindex(Xbl) ; Tbl[k] = similar(Xbl[1], n, nlv) ; end
    Tb = list(Matrix{Q}, nlv)
    for a = 1:nlv ; Tb[a] = similar(Xbl[1], n, nbl) ; end
    Vbl = list(Matrix{Q}, nbl)
    for k in eachindex(Xbl) ; Vbl[k] = similar(Xbl[1], nco(Xbl[k]), nlv) ; end
    u = similar(Xbl[1], n)
    tk = copy(u)
    w = similar(Xbl[1], nbl)
    lb = similar(Xbl[1], nbl, nlv)
    mu = similar(Xbl[1], nlv)
    niter = zeros(nlv)
    # End
    res = 0
    @inbounds for a = 1:nlv
        X = reduce(hcat, Xbl)
        u .= X[:, 1]
        #u .= nipals(X).u  # makes niter = 1
        iter = 1
        cont = true
        while cont
            u0 = copy(u)
            for k in eachindex(Xbl)
                vk = Xbl[k]' * u    # = vktild
                dk = normv(vk)
                vk ./= dk           # vk is normed
                tk .= Xbl[k] * vk 
                Tb[a][:, k] .= tk
                Tbl[k][:, a] .= fweight(Tb[a][:, k], invsqrtw)
                Vbl[k][:, a] .= vk
                lb[k, a] = dk^2
            end
            res = nipals(Tb[a])
            u .= res.u 
            w .= res.v
            dif = sum((u .- u0).^2)
            iter = iter + 1
            if (dif < par.tol) || (iter > par.maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        U[:, a] .= u .* invsqrtw
        W[:, a] .= w
        mu[a] = res.sv^2  # = sum(lb)
        for k in eachindex(Xbl)
            Xbl[k] .-= u * (u' * Xbl[k])
        end
    end
    T = sqrt.(mu)' .* U    
    Mbpca(T, U, W, Tb, Tbl, Vbl, lb, mu, fitmbl, weights, niter, par)
end

""" 
    transf(object::Mbpca, Xbl; nlv = nothing)
    transfbl(object::Mbpca, Xbl; nlv = nothing)
Compute latent variables (LVs = scores T) from 
    a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transf(object::Mbpca, Xbl; nlv = nothing)
    transf_all(object, Xbl; nlv).T
end

function transfbl(object::Mbpca, Xbl; nlv = nothing)
    transf_all(object, Xbl; nlv).Tbl
end

function transf_all(object::Mbpca, Xbl; nlv = nothing)
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
        u .= 1 / sqrt(object.mu[a]) * TB * object.W[:, a]
        U[:, a] .= u
        @inbounds for k in eachindex(Xbl)
            Vx = sqrt(object.lb[k, a]) * object.Vbl[k][:, a]'
            zXbl[k] -= u * Vx
        end
    end
    T = sqrt.(object.mu[1:nlv])' .* U
    (T = T, Tbl)
end

"""
    summary(object::Mbpca, Xbl)
Summarize the fitted model.
* `object` : The fitted model.
* `Xbl` : The X-data that was used to 
    fit the model.
""" 
function Base.summary(object::Mbpca, Xbl)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)
    nlv = nco(object.T)
    sqrtw = sqrt.(object.weights.w)
    zXbl = transf(object.fitmbl, Xbl)
    @inbounds for k in eachindex(Xbl)
        fweight!(zXbl[k], sqrtw)
    end
    X = fconcat(zXbl)
    ## Proportion of the X-inertia explained by each global LV
    sstot = zeros(Q, nbl)
    @inbounds for k in eachindex(Xbl)
        sstot[k] = frob2(zXbl[k])
    end
    tt = colsum(object.lb)    
    pvar = tt / sum(sstot)
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(lv = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    ## Within each block, proportion of the block-inertia explained by each global LV
    ## = object.lb if bscal = :frob 
    z = fscale(object.lb', sstot)'
    nam = string.("lv", 1:nlv)
    explX = DataFrame(z, nam)
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
    (explvarx = explvarx, explX, contr_block, cortbl2t, corx2t, rv = zrv, lg = zlg)
end



