"""
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
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling. See function `blockscal`
    for possible values.
* `tol` : Tolerance value for convergence (Nipals).
* `maxit` : Maximum number of iterations (Nipals).
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    and `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

This functions implements the MBPLSR Nipals algorithm such 
as in Westerhuis et al. 1998. The function gives the same 
results as function `mbplsr`.

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
model = mbplswest; nlv, bscal, scal)
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
res.corx2t 
res.cortb2t 
res.rdx
```
"""
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
    @inbounds for k = 1:nbl
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
    sqrtw = sqrt.(weights.w)
    fitmbl = blockscal(Xbl, weights; bscal = par.bscal, centr = true, scal = par.scal)
    transf!(fitmbl, Xbl)
    X = reduce(hcat, Xbl)
    ymeans = colmean(Y, weights)
    yscales = ones(Q, q)
    if par.scal 
        yscales .= colstd(Y, weights)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(Y, ymeans)
    end
    # Row metric
    p = list(Int, nbl)
    @inbounds for k = 1:nbl
        p[k] = nco(Xbl[k])
        Xbl[k] .= sqrtw .* Xbl[k]
    end
    Y .= sqrtw .* Y
    ## Pre-allocation
    X = similar(Xbl[1], n, sum(p))
    Tbl = list(Matrix{Q}, nbl)
    for k = 1:nbl ; Tbl[k] = similar(Xbl[1], n, nlv) ; end
    Tb = list(Matrix{Q}, nlv)
    for a = 1:nlv ; Tb[a] = similar(Xbl[1], n, nbl) ; end
    Pbl = list(Matrix{Q}, nbl)
    for k = 1:nbl ; Pbl[k] = similar(Xbl[1], nco(Xbl[k]), nlv) ; end
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
    niter = zeros(nlv)
    # End
    @inbounds for a = 1:nlv
        ty = Y[:, 1]
        cont = true
        iter = 1
        while cont
            t0 = copy(ty)
            for k = 1:nbl
                wktild = Xbl[k]' * ty / dot(ty, ty)
                dk = norm(wktild)
                wk = wktild / dk
                tk = Xbl[k] * wk
                pk =  Xbl[k]' * tk
                pk ./= dot(tk, tk)
                Pbl[k][:, a] .= pk
                Tb[a][:, k] .= tk
                Tbl[k][:, a] .= (1 ./ sqrtw) .* tk  
            end
            w = Tb[a]' * ty / dot(ty, ty) 
            w ./= norm(w)
            tx .= Tb[a] * w
            wy .= Y' * tx
            wy ./= norm(wy)
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
        end
        Y .-= tx * wytild'
    end
    Tx .= (1 ./ sqrtw) .* Tx
    Rx = Wx * inv(Px' * Wx)
    lb = nothing
    Mbplswest(Tx, Px, Rx, Wx, Wytild, Tbl, Tb, Pbl, TTx, fitmbl, ymeans, yscales, 
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
    sqrtw = sqrt.(object.weights.w)
    zXbl = transf(object.fitmbl, Xbl)
    @inbounds for k = 1:nbl
        zXbl[k] .= sqrtw .* zXbl[k]
    end
    X = reduce(hcat, zXbl)
    # Explained_X
    ssk = zeros(Q, nbl)
    @inbounds for k = 1:nbl
        ssk[k] = ssq(zXbl[k])
    end
    sstot = sum(ssk)
    tt = object.TT
    tt_adj = colsum(object.P.^2) .* tt
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)     
    ## Correlation between the original X-variables
    ## and the global scores
    z = cor(X, sqrtw .* object.T)  
    corx2t = DataFrame(z, string.("lv", 1:nlv))
    ## Correlation between the X-block scores 
    ## and the global scores 
    z = list(Matrix{Q}, nlv)
    @inbounds for a = 1:nlv
        z[a] = cor(object.Tb[a], sqrtw .* object.T[:, a])
    end
    cortb2t = DataFrame(reduce(hcat, z), string.("lv", 1:nlv))
    ## Redundancies (Average correlations) Rd(X, t) 
    ## between each X-block and each global score
    z = list(Matrix{Q}, nbl)
    @inbounds for k = 1:nbl
        z[k] = rd(zXbl[k], sqrtw .* object.T)
    end
    rdx = DataFrame(reduce(vcat, z), string.("lv", 1:nlv))         
    ## Specific weights of each block on 
    ## each X-global score
    #sal2 = nothing
    #if !isnothing(object.lb)
    #    lb2 = colsum(object.lb.^2)
    #    sal2 = fscale(object.lb.^2, lb2)
    #    sal2 = DataFrame(sal2, 
    #        string.("lv", 1:nlv))
    #end
    ## End
    (explvarx = explvarx, corx2t, cortb2t, rdx)
end
