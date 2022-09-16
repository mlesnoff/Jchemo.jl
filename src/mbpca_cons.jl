struct MbpcaCons
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
    mbpca_cons(X_bl, weights = ones(size(X_bl[1], 1)); nlv,
        bscal = "none", tol = sqrt(eps(1.)), maxit = 200)
Consensus principal components analysis (= CPCA, MBPCA).
* `X_bl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `weights` : Weights of the observations (rows). 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscal` : Type of block scaling (`"none"`, `"frob"`, `"mfa"`). 
    See functions `blockscal`.
* `tol` : Tolerance value for convergence.
* `niter` : Maximum number of iterations.

`weights` is internally normalized to sum to 1.

The function returns several objects, in particular:
* `T` : The non normed global scores.
* `U` : The normed global scores.
* `W` : The global loadings.
* `Tb` : The block scores.
* `W_bl` : The block loadings.
* `lb` : The specific weights "lambda".
* `mu` : The sum of the specific weights (= eigen value of the global PCA).

Function `summary` returns: 
* `explvarx` : Proportion of the total inertia of X (sum of the squared norms of the 
    blocks) explained by each global score.
* `contr_block` : Contribution of each block to the global scores 
* `explX` : Proportion of the inertia of the blocks explained by each global score.
* `cort2x` : Correlation between the global scores and the original variables.  
* `cort2tb` : Correlation between the global scores and the block scores.
* `rv` : RV coefficient. 
* `lg` : Lg coefficient. 

## References
Tchandao Mangamana, E., Cariou, V., Vigneau, E., Glèlè Kakaï, R.L., Qannari, E.M., 2019. 
Unsupervised multiblock data analysis: A unified approach and extensions. Chemometrics and 
Intelligent Laboratory Systems 194, 103856. https://doi.org/10.1016/j.chemolab.2019.103856

Westerhuis, J.A., Kourti, T., MacGregor, J.F., 1998. Analysis of multiblock and hierarchical 
PCA and PLS models. Journal of Chemometrics 12, 301–321. 
https://doi.org/10.1002/(SICI)1099-128X(199809/10)12:5<301::AID-CEM515>3.0.CO;2-S

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
X_bl = mblock(X, listbl)
# "New" = first two rows of X_bl 
X_bl_new = mblock(X[1:2, :], listbl)

bscal = "frob"
fm = mbpca_cons(X_bl; nlv = 4, bscal = bscal) ;
fm.U
fm.T
Jchemo.transform(fm, X_bl)
Jchemo.transform(fm, X_bl_new) 

res = Jchemo.summary(fm, X_bl) ;
fm.lb
rowsum(fm.lb)
fm.mu
res.explvarx
res.explX # = fm.lb if bscal = "frob"
rowsum(Matrix(res.explX))
res.contr_block
res.cort2x 
res.cort2tb
res.rv
```
"""
function mbpca_cons(X_bl, weights = ones(size(X_bl[1], 1)); nlv,
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
    if bscal == "mfa"
        res = blockscal_mfa(X, weights) 
        scal = res.scal
        X = res.X
    end
    # Row metric
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
    W = similar(X[1], nbl, nlv)
    w = similar(X[1], nbl)
    mu = similar(X[1], nlv)
    niter = zeros(nlv)
    # End
    for a = 1:nlv
        zX = reduce(hcat, X)
        u .= nipals(zX).u
        iter = 1
        cont = true
        while cont
            u0 = copy(u)
            for k = 1:nbl
                wb = X[k]' * u
                wb ./= norm(wb)
                tb .= X[k] * wb 
                Tb[a][:, k] .= tb
                lb[k, a] = dot(tb, u)^2    
                W_bl[k][:, a] .= wb
            end
            res = nipals(Tb[a])
            u .= res.u 
            w .= res.v
            dif = sum((u .- u0).^2)
            iter = iter + 1
            if (dif < tol) || (iter > maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        U[:, a] .= u
        W[:, a] .= w
        mu[a] = res.sv^2  # = sum(lb)
        for k = 1:nbl
            X[k] .-= u * (u' * X[k])
        end
    end
    T = Diagonal(1 ./ sqrtw) * (sqrt.(mu)' .* U)
    MbpcaCons(T, U, W, Tb, W_bl, lb, mu,
        xmeans, scal, weights, niter)
end

""" 
    transform(object::MbpcaCons, X_bl; nlv = nothing)
Compute components (scores matrix "T") from a fitted model and X-data.
* `object` : The maximal fitted model.
* `X_bl` : A list (vector) of blocks (matrices) of X-data for which LVs are computed.
* `nlv` : Nb. components to compute. If nothing, it is the maximum nb. PCs.
""" 
function transform(object::MbpcaCons, X_bl; nlv = nothing)
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
    summary(object::MbpcaCons, X_bl)
Summarize the fitted model.
* `object` : The fitted model.
* `X_bl` : The X-data that was used to fit the model.
""" 
function summary(object::MbpcaCons, X_bl)
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
    (explvarx = explvarx, contr_block, explX, 
        cort2x, cort2tb, rv = zrv, lg = zlg)
end







