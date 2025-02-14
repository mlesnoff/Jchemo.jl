"""
    mbplswest(; kwargs...)
    mbplswest(Xbl, Y; kwargs...)
    mbplswest(Xbl, Y, weights::Weight; kwargs...)
    mbplswest!(Xbl::Matrix, Y::Matrix, weights::Weight; kwargs...)
Multiblock PLSR (MBPLSR) - Nipals algorithm.
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. global latent variables (LVs = scores) to compute.
* `bscal` : Type of block scaling. See function `blockscal`
    for possible values.
* `tol` : Tolerance value for convergence (Nipals).
* `maxit` : Maximum number of iterations (Nipals).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    and `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

This functions implements the MBPLSR Nipals algorithm such 
as in Westerhuis et al. 1998. The function gives the same 
global scores and predictions as function `mbplsr`.

Function `summary` returns: 
* `explvarx` : Proportion of the total X inertia (squared Frobenious norm) 
    explained by the global LVs.
* `rdxbl2t` : Rd coefficients between each block (= Xbl[k]) and the global LVs.
* `rvxbl2t` : RV coefficients between each block and the global LVs.
* `cortbl2t` : Correlations between the block LVs (= Tbl[k]) and the global LVs.
* `corx2t` : Correlation between the X-variables and the global LVs.  

## References 
Westerhuis, J.A., Kourti, T., MacGregor, J.F., 1998. Analysis 
of multiblock and hierarchical PCA and PLS models. Journal of 
Chemometrics 12, 301–321. 
https://doi.org/10.1002/(SICI)1099-128X(199809/10)12:5<301::AID-CEM515>3.0.CO;2-S

## Examples
```julia
using Jchemo, JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "ham.jld2") 
@load db dat
pnames(dat) 
X = dat.X
Y = dat.Y
y = Y.c1
group = dat.group
listbl = [1:11, 12:19, 20:25]
s = 1:6
Xbltrain = mblock(X[s, :], listbl)
Xbltest = mblock(rmrow(X, s), listbl)
ytrain = y[s]
ytest = rmrow(y, s) 
ntrain = nro(ytrain) 
ntest = nro(ytest) 
ntot = ntrain + ntest 
(ntot = ntot, ntrain , ntest)

nlv = 3
bscal = :frob
scal = false
#scal = true
model = mbplswest(; nlv, bscal, scal)
fit!(model, Xbltrain, ytrain)
pnames(model) 
pnames(model.fitm)
@head model.fitm.T
@head transf(model, Xbltrain)
transf(model, Xbltest)

res = predict(model, Xbltest)
res.pred 
rmsep(res.pred, ytest)

res = summary(model, Xbltrain) ;
pnames(res) 
res.explvarx
res.rdxbl2t
res.rvxbl2t
res.cortbl2t
res.corx2t 
```
"""
mbplswest(; kwargs...) = JchemoModel(mbplswest, nothing, kwargs)

function mbplswest(Xbl, Y; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    weights = mweight(ones(Q, n))
    mbplswest(Xbl, Y, weights; kwargs...)
end

function mbplswest(Xbl, Y, weights::Weight; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)  
    zXbl = list(Matrix{Q}, nbl)
    @inbounds for k in eachindex(Xbl)
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    mbplswest!(zXbl, copy(ensure_mat(Y)), weights; kwargs...)
end

function mbplswest!(Xbl::Vector, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParMbplsr, kwargs).par
    @assert in([:none, :frob])(par.bscal) "Wrong value for argument 'bscal'."
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)
    n = nro(Xbl[1])
    q = nco(Y)
    nlv = par.nlv
    ## Block scaling
    fitmbl = blockscal(Xbl, weights; centr = true, scal = par.scal, bscal = par.bscal)
    transf!(fitmbl, Xbl)
    X = reduce(hcat, Xbl)
    ## Y centering/scaling
    ymeans = colmean(Y, weights)
    yscales = ones(Q, q)
    if par.scal 
        yscales .= colstd(Y, weights)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(Y, ymeans)
    end
    # Row metric
    sqrtw = sqrt.(weights.w)
    invsqrtw = 1 ./ sqrtw
    p = zeros(Int, nbl)
    @inbounds for k in eachindex(Xbl) 
        p[k] = nco(Xbl[k])
        fweight!(Xbl[k], sqrtw)
    end
    fweight!(Y, sqrtw)
    ## Pre-allocation
    X = similar(Xbl[1], n, sum(p))
    Tbl = list(Matrix{Q}, nbl)
    for k in eachindex(Xbl) ; Tbl[k] = similar(Xbl[1], n, nlv) ; end
    Tb = list(Matrix{Q}, nlv)
    for a = 1:nlv ; Tb[a] = similar(Xbl[1], n, nbl) ; end
    Pbl = list(Matrix{Q}, nbl)
    for k in eachindex(Xbl) ; Pbl[k] = similar(Xbl[1], nco(Xbl[k]), nlv) ; end
    Tx = similar(Xbl[1], n, nlv)
    Wx = similar(Xbl[1], sum(p), nlv)
    Wytild = similar(Xbl[1], q, nlv)
    Vx = copy(Wx)
    tk  = similar(Xbl[1], n)
    tx = copy(tk)
    ty  = copy(tk)
    wx = similar(Xbl[1], sum(p))
    vx = copy(wx)
    wy  = similar(Xbl[1], q)
    wytild = copy(wy)
    TTx = similar(Xbl[1], nlv)
    niter = zeros(nlv)
    # End
    @inbounds for a = 1:nlv
        ty = Y[:, 1]
        cont = true
        iter = 1
        while cont
            t0 = copy(ty)
            for k in eachindex(Xbl)
                wktild = Xbl[k]' * ty / dot(ty, ty)
                dk = normv(wktild)
                wk = wktild / dk
                tk = Xbl[k] * wk
                pk =  Xbl[k]' * tk
                pk ./= dot(tk, tk)
                Pbl[k][:, a] .= pk
                Tb[a][:, k] .= tk
                Tbl[k][:, a] .= invsqrtw .* tk  
            end
            w = Tb[a]' * ty / dot(ty, ty) 
            w ./= normv(w)
            tx .= Tb[a] * w
            wy .= Y' * tx
            wy ./= normv(wy)
            ty .= Y * wy
            dif = sum((ty .- t0).^2)
            iter = iter + 1
            if (dif < par.tol) || (iter > par.maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        # For global
        ttx = dot(tx, tx)
        X .= fconcat(Xbl)
        wx .= X' * ty / dot(ty, ty)    
        wx ./= normv(wx)
        mul!(vx, X', tx)
        vx ./= ttx
        wytild .= Y' * tx / ttx
        # End           
        Tx[:, a] .= tx   
        Wx[:, a] .= wx
        Vx[:, a] .= vx
        Wytild[:, a] .= wytild
        TTx[a] = ttx
        @inbounds for k in eachindex(Xbl)
            Xbl[k] .-= tx * tx' * Xbl[k] / ttx
        end
        Y .-= tx * wytild'
    end
    fweight!(Tx, invsqrtw)
    Rx = Wx * inv(Vx' * Wx)
    lb = nothing
    Mbplswest(Tx, Vx, Rx, Wx, Wytild, Tb, Tbl, Pbl, TTx, fitmbl, ymeans, yscales, 
        weights, lb, niter, par)
end

"""
    summary(object::Mbplswest, Xbl)
Summarize the fitted model.
* `object` : The fitted model.
* `Xbl` : The X-data that was used to 
    fit the model.
""" 
function Base.summary(object::Mbplswest, Xbl)
    Q = eltype(Xbl[1][1, 1])
    n, nlv = size(object.T)
    nbl = length(Xbl)
    ## Block scaling
    zXbl = transf(object.fitmbl, Xbl)
    X = fconcat(zXbl)
    ## Proportion of the total X-inertia explained by each global LV
    ssk = zeros(Q, nbl)
    @inbounds for k in eachindex(Xbl)
        ssk[k] = frob2(zXbl[k], object.weights)
    end
    tt = object.TT
    tt_adj = (colnorm(object.V).^2) .* tt  # tt_adj[a] = p[a]'p[a] * tt[a]
    pvar = tt_adj / sum(ssk)
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## Rd between each Xk and the global LVs
    nam = string.("lv", 1:nlv)
    z = zeros(Q, nbl, nlv)
    for k in eachindex(Xbl) 
        z[k, :] = rd(zXbl[k], object.T, object.weights) 
    end
    rdxbl2t = DataFrame(z, nam)
    ## RV between each Xk and the global LVs
    z = zeros(Q, nbl, nlv)
    for k in eachindex(Xbl), a = 1:nlv
        z[k, a] = rv(zXbl[k], object.T[:, a], object.weights) 
    end
    rvxbl2t = DataFrame(z, nam)
    ## Correlation between the block LVs and the global LVs
    z = zeros(Q, nbl, nlv)
    for k in eachindex(Xbl), a = 1:nlv 
        z[k, a] = corv(object.Tbl[k][:, a], object.T[:, a], object.weights) 
    end
    cortbl2t = DataFrame(z, nam)
    ## Correlation between the X-variables and the global LVs 
    z = corm(X, object.T, object.weights)  
    corx2t = DataFrame(z, nam)  
    (explvarx = explvarx, rdxbl2t, rvxbl2t, cortbl2t, corx2t)
end
