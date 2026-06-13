"""
    cpca(; kwargs...)
    cpca(Xbl; kwargs...)
    cpca(Xbl, weights::ProbabilityWeights; kwargs...)
    cpca!(Xbl::Matrix, weights::ProbabilityWeights; kwargs...)
Consensus principal components analysis (CPCA, a.k.a MBPCA) by Nipals.
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock`.  
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. global latent variables (LVs; = scores) to compute.
* `bscal` : Type of block scaling. See function `blockscal` for possible values.
* `tol` : Tolerance value for Nipals convergence.
* `maxit` : Maximum number of iterations (Nipals).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` is scaled by its uncorrected standard 
    deviation (before the block scaling).

CPCA Nipals algorithm (Westerhuis et a; 1998), also known as MBPCA, and referred to as CPCA-W in Smilde et al. 2003. 
Besides an eventual block scaling, CPCA is equivalent to a PCA on the horizontally concatenated matrix X = [X1 X2 ... Xk],
referred to as SUM-PCA in Smilde et al 2003.

The function returns several objects, in particular:
* `T` : Global LVs (not-normed).
* `U` : Global LVs (normed).
* `W` : Block weights (normed).
* `Tb` : Block LVs (in the metric scale), returned **grouped by LV**.
* `Tbl` : Block LVs (in the original scale), returned **grouped by block**.
* `Vbl` : Block loadings (normed).
* `lb` : Block specific weights ('lambda') for the global LVs.
* `mu` : Sum of the block specific weights (= eigen values of the global PCA).

Function `summary` returns: 
* `explvarx` : Proportion of the total X inertia (squared Frobenious norm) explained by the global LVs.
* `explxbl` : Proportion of the inertia of each block (= Xbl[k]) explained by the global LVs.
* `contrxbl2t` : Contribution of each block to the global LVs (= lb proportions).  
* `rvxbl2t` : RV coefficients between each block and the global LVs.
* `rdxbl2t` : Rd coefficients between each block and the global LVs.
* `cortbl2t` : Correlations between the block LVs (= Tbl[k]) and the global LVs.
* `corx2t` : Correlation between the X-variables and the global LVs.  

## References
Mangamana, E.T., Cariou, V., Vigneau, E., Glèlè Kakaï, R.L., Qannari, E.M., 2019. Unsupervised multiblock data 
analysis: A unified approach and extensions. Chemometrics and Intelligent Laboratory Systems 194, 103856. 
https://doi.org/10.1016/j.chemolab.2019.103856

Smilde, A.K., Westerhuis, J.A., de Jong, S., 2003. A framework for sequential multiblock component methods. 
Journal of Chemometrics 17, 323–337. https://doi.org/10.1002/cem.811

Westerhuis, J.A., Kourti, T., MacGregor, J.F., 1998. Analysis of multiblock and hierarchical PCA and PLS models. Journal 
of Chemometrics 12, 301–321. https://doi.org/10.1002/(SICI)1099-128X(199809/10)12:5<301::AID-CEM515>3.0.CO;2-S

## Examples
```julia
using Jchemo, JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "ham.jld2") 
@load db dat
@names dat 
X = dat.X
group = dat.group
listbl = [1:11, 12:19, 20:25]
Xbl = mblock(X[1:6, :], listbl)
Xblnew = mblock(X[7:8, :], listbl)
n = nro(Xbl[1]) 

nlv = 3
bscal = :frob
#bscal = :none
scal = false
#scal = true
model = cpca(; nlv, bscal, scal, tol = 1e-15)
fit!(model, Xbl)
@names model 
fitm = model.fitm ;
@names fitm

## Global scores 
@head transf(model, Xbl)
@head fitm.T

transf(model, Xblnew)

## Blocks scores
i = 1
@head transfbl(model, Xbl)[i]
@head fitm.Tbl[i]

## Summary
res = summary(model, Xbl) ;
@names res 
res.explvarx
res.explxbl   # = fitm.lb if bscal = :frob
rowsum(Matrix(res.explxbl))
res.contrxbl2t
res.rvxbl2t
res.rdxbl2t
res.cortbl2t
res.corx2t 

#### This CPCA can also be implemented with function 'pip'

model1 = blockscal(; bscal, centr = true) ;
model2 = mbconcat()
model3 = pcasvd(; nlv) ;
model = pip(model1, model2, model3)
fit!(model, Xbl)

mod3 = model.model[3] ;
typeof(mod3) 
@names mod3 
@names mod3.fitm

@head transf(model, Xbl)
@head mod3.fitm.T 

transf(model, Xblnew)

#### And a sparse CPCA as follows

meth = :soft ; nvar = 2
model1 = blockscal(; bscal, centr = true) ;
model2 = mbconcat()
model3 = spca(; nlv, meth, nvar) ;
model = pip(model1, model2, model3)
fit!(model, Xbl)

mod3 = model.model[3] ;
@names mod3 
typeof(mod3) 
@names mod3.fitm

mod3.fitm.sellv
mod3.fitm.sel

@head transf(model, Xbl)
@head mod3.fitm.T 

transf(model, Xblnew)
```
"""
cpca(; kwargs...) = JchemoModel(cpca, nothing, kwargs)

function cpca(Xbl; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    cpca(Xbl, pweight(ones(Q, n)); kwargs...)
end

function cpca(Xbl, weights::ProbabilityWeights; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)  
    zXbl = list(Matrix{Q}, nbl)
    @inbounds for k in eachindex(Xbl)
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    cpca!(zXbl, weights; kwargs...)
end

function cpca!(Xbl::Vector, weights::ProbabilityWeights; kwargs...)
    par = recovkw(ParCpca, kwargs).par 
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    nbl = length(Xbl)
    pbl = nco.(Xbl) ; ptot = sum(pbl)
    nlv = min(n, ptot, par.nlv) # to do: consider if ptot should not be replaced by pmin = minimum(pbl)
    par.nlv = nlv
    ## Block scaling
    fitm_bl = blockscal(Xbl, weights; centr = true, scal = par.scal, bscal = par.bscal)
    transf!(fitm_bl, Xbl)
    # Row metric
    sqrtw = sqrt.(weights.values)
    invsqrtw = 1 ./ sqrtw
    @inbounds for k in eachindex(Xbl) 
        fweightr!(Xbl[k], sqrtw)
    end
    ## Pre-allocation
    U = similar(Xbl[1], n, nlv)
    W = similar(Xbl[1], nbl, nlv)
    Tbl = list(Matrix{Q}, nbl)
    for k in eachindex(Xbl) ; Tbl[k] = similar(Xbl[1], n, nlv) ; end
    Tb = list(Matrix{Q}, nlv)
    for a = 1:nlv ; Tb[a] = similar(Xbl[1], n, nbl) ; end
    Vbl = list(Matrix{Q}, nbl)
    for k in eachindex(Xbl) ; Vbl[k] = similar(Xbl[1], pbl[k], nlv) ; end
    u = similar(Xbl[1], n)
    tk = similar(u)
    w = similar(Xbl[1], nbl)
    lb = similar(Xbl[1], nbl, nlv)
    mu = similar(Xbl[1], nlv)
    niter = zeros(nlv)
    # End
    res = 0
    @inbounds for a = 1:nlv
        X = fconcat(Xbl)
        u .= X[:, 1]
        #u .= nipals(X).u  # forces niter = 1
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
                Tbl[k][:, a] .= fweightr(Tb[a][:, k], invsqrtw)
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
        @. U[:, a] = u * invsqrtw
        W[:, a] .= w
        mu[a] = res.sv^2  # = sum(lb)
        for k in eachindex(Xbl)
            Xbl[k] .-= u * (u' * Xbl[k])
        end
    end
    T = sqrt.(mu)' .* U    
    Cpca(T, U, W, Tb, Tbl, Vbl, lb, mu, fitm_bl, weights, niter, par)
end

""" 
    transf(object::Cpca, Xbl)
    transf(object::Cpca, Xbl, nlv::Int)
    transfbl(object::Cpca, Xbl)
    transfbl(object::Cpca, Xbl, nlv::Int)
Compute latent variables (LVs; = scores) from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) of X-data for which LVs are computed.
* `nlv` : Nb. LVs to compute.
""" 

transf(object::Cpca, Xbl) = transf_all(object, Xbl, object.par.nlv).T
transf(object::Cpca, Xbl, nlv::Int) = transf_all(object, Xbl, nlv).T

transfbl(object::Cpca, Xbl) = transf_all(object, Xbl, object.par.nlv).Tbl
transfbl(object::Cpca, Xbl, nlv::Int) = transf_all(object, Xbl, nlv).Tbl

function transf_all(object::Cpca, Xbl, nlv::Int)
    Q = eltype(Xbl[1][1, 1])
    a = object.par.nlv
    nlv = isnothing(nlv) ? a : min(nlv, a)
    nbl = length(Xbl)
    m = size(Xbl[1], 1)
    zXbl = transf(object.fitm_bl, Xbl)
    U = similar(zXbl[1], m, nlv)
    TB = similar(zXbl[1], m, nbl)
    Tbl = list(Matrix{Q}, nbl)
    for k in eachindex(Xbl) ; Tbl[k] = similar(zXbl[1], m, nlv) ; end
    u = similar(zXbl[1], m)
    tk = similar(u)
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
    summary(object::Cpca, Xbl)
Summarize the fitted model.
* `object` : The fitted model.
* `Xbl` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Cpca, Xbl)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)
    nlv = nco(object.T)
    ## Block scaling
    zXbl = transf(object.fitm_bl, Xbl)
    X = fconcat(zXbl)
    ## Proportion of the total X-inertia explained by each global LV
    ssk = zeros(Q, nbl)
    @inbounds for k in eachindex(Xbl)
        ssk[k] = frob2(zXbl[k], object.weights)
    end
    tt = colsum(object.lb)    
    pvar = tt / sum(ssk)
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(lv = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    ## Within each block k, proportion of the Xk-inertia explained by the global LVs
    ## = object.lb if bscal = :frob 
    nam = string.("lv", 1:nlv)
    z = fscale(object.lb', ssk)'
    explxbl = DataFrame(z, nam)
    ## Contribution of each block Xk to the global LVs = lb proportions
    z = fscale(object.lb, colsum(object.lb))
    contrxbl2t = DataFrame(z, nam)
    ## RV between each Xk and the global LVs
    z = zeros(Q, nbl, nlv)
    for k in eachindex(Xbl), a = 1:nlv
        z[k, a] = rv(zXbl[k], object.T[:, a], object.weights) 
    end
    rvxbl2t = DataFrame(z, nam)
    ## Rd between each Xk and the global LVs
    z = zeros(Q, nbl, nlv)
    for k in eachindex(Xbl) 
        z[k, :] = rd(zXbl[k], object.T, object.weights) 
    end
    rdxbl2t = DataFrame(z, nam)
    ## Correlation between the block LVs and the global LVs
    z = zeros(Q, nbl, nlv)
    for k in eachindex(Xbl), a = 1:nlv 
        z[k, a] = corv(object.Tbl[k][:, a], object.T[:, a], object.weights) 
    end
    cortbl2t = DataFrame(z, nam)
    ## Correlation between the X-variables and the global LVs 
    z = corm(X, object.T, object.weights)  
    corx2t = DataFrame(z, nam)  
    (explvarx = explvarx, explxbl, contrxbl2t, rvxbl2t, rdxbl2t, cortbl2t, corx2t) 
end



