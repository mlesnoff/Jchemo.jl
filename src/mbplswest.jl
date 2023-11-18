"""
    mbplswest(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = :none, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
    mbplswest!(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = :none, tol = sqrt(eps(1.)), maxit = 200, 
        scal::Bool = false)
Multiblock PLSR - Nipals algorithm (Westerhuis et al. 1998).
* `Xbl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `Y` : Y-data.
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscal` : Type of `Xbl` block scaling (`:none`, `:frob`). 
    See functions `blockscal`.
* `tol` : Tolerance value for convergence.
* `maxit` : Maximum number of iterations.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` and 
    of `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

MBPLSR is equivalent to the the PLSR (X, `Y`) where X is the horizontal 
concatenation of the blocks in `Xbl`.
The function gives the same results as function `mbplsr`.

## References 
Westerhuis, J.A., Kourti, T., MacGregor, J.F., 1998. Analysis of multiblock and hierarchical 
PCA and PLS models. Journal of Chemometrics 12, 301–321. 
https://doi.org/10.1002/(SICI)1099-128X(199809/10)12:5<301::AID-CEM515>3.0.CO;2-S

## Examples
```julia
using JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/ham.jld2") 
@load db dat
pnames(dat) 

X = dat.X
Y = dat.Y
y = dat.Y.c1
group = dat.group
listbl = [1:11, 12:19, 20:25]
Xbl = mblock(X, listbl)
# "New" = first two rows of Xbl 
Xbl_new = mblock(X[1:2, :], listbl)

bscal = :none
nlv = 5
fm = mbplswest(Xbl, y; nlv = nlv, bscal = bscal) ;
pnames(fm)
fm.T
Jchemo.transform(fm, Xbl_new)
[y Jchemo.predict(fm, Xbl).pred]
Jchemo.predict(fm, Xbl_new).pred

summary(fm, Xbl)
```
"""
function mbplswest(Xbl, Y; par = Par())
    T = eltype(Xbl[1])
    n = nro(Xbl[1])
    weights = mweight(ones(T, n))
    mbplswest(Xbl, Y, weights; par)
end

function mbplswest(Xbl, Y, weights::Weight; par = Par())
    nbl = length(Xbl)  
    zXbl = list(nbl, Matrix)
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    mbplswest!(zXbl, copy(ensure_mat(Y)), 
        weights; par)
end

function mbplswest!(Xbl, Y::Matrix, weights::Weight; 
        par = Par())
    @assert in([:none; :frob])(par.bscal) "Wrong value for argument 'bscal'."
    nbl = length(Xbl)
    n = nro(Xbl[1])
    q = nco(Y)
    nlv = par.nlv
    Q = eltype(Xbl[1])
    sqrtw = sqrt.(weights.w)
    xmeans = list(nbl, Vector)
    xscales = list(nbl, Vector)
    p = fill(0, nbl)
    Threads.@threads for k = 1:nbl
        p[k] = nco(Xbl[k])
        xmeans[k] = colmean(Xbl[k], weights) 
        xscales[k] = ones(Q, nco(Xbl[k]))
        if par.scal 
            xscales[k] = colstd(Xbl[k], weights)
            Xbl[k] .= cscale(Xbl[k], xmeans[k], xscales[k])
        else
            Xbl[k] .= center(Xbl[k], xmeans[k])
        end
    end
    ymeans = colmean(Y, weights)
    yscales = ones(Q, q)
    if par.scal 
        yscales .= colstd(Y, weights)
        cscale!(Y, ymeans, yscales)
    else
        center!(Y, ymeans)
    end
    par.bscal == :none ? bscales = ones(Q, nbl) : nothing
    if par.bscal == :frob
        res = blockscal_frob(Xbl, weights) 
        bscales = res.bscales
        Xbl = res.Xbl
    end
    # Row metric
    @inbounds for k = 1:nbl
        Xbl[k] .= sqrtw .* Xbl[k]
    end
    Y .= sqrtw .* Y
    # Pre-allocation
    X = similar(Xbl[1], n, sum(p))
    Tbl = list(nbl, Matrix)
    for k = 1:nbl ; Tbl[k] = similar(Xbl[1], n, nlv) ; end
    Tb = list(nlv, Matrix)
    for a = 1:nlv ; Tb[a] = similar(Xbl[1], n, nbl) ; end
    Pbl = list(nbl, Matrix)
    for k = 1:nbl ; Pbl[k] = similar(Xbl[1], p[k], nlv) ; end
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
    MbplsWest(Tx, Px, Rx, Wx, Wytild, Tbl, Tb, Pbl, TTx,    
        bscales, xmeans, xscales, ymeans, yscales, weights, lb, niter)
end

""" 
    transform(object::Union{MbplsWest, Mbplsr}, Xbl; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list (vector) of blocks (matrices) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transform(object::Union{MbplsWest, Mbplsr}, Xbl; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    nbl = length(Xbl)
    zXbl = list(nbl, Matrix{eltype(Xbl[1])})
    Threads.@threads for k = 1:nbl
        zXbl[k] = cscale(Xbl[k], object.xmeans[k], object.xscales[k])
    end
    res = blockscal(zXbl, object.bscales)
    reduce(hcat, res.Xbl) * vcol(object.R, 1:nlv) 
end

"""
    predict(object::Union{MbplsWest, Mbplsr}, Xbl; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list (vector) of X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Union{MbplsWest, Mbplsr}, Xbl; 
        nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = (max(0, minimum(nlv)):min(a, maximum(nlv)))
    le_nlv = length(nlv)
    T = transform(object, Xbl)
    pred = list(le_nlv, Matrix{eltype(Xbl[1])})
    @inbounds  for i = 1:le_nlv
        znlv = nlv[i]
        int = object.ymeans'
        B = object.C[:, 1:znlv]'
        pred[i] = int .+ vcol(T, 1:znlv) * B 
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end

"""
    summary(object::MbplsWest, Xbl)
Summarize the fitted model.
* `object` : The fitted model.
* `Xbl` : The X-data that was used to fit the model.
""" 
function Base.summary(object::MbplsWest, Xbl)
    n, nlv = size(object.T)
    nbl = length(Xbl)
    sqrtw = sqrt.(object.weights.w)
    zXbl = list(nbl, Matrix{eltype(Xbl[1])})
    Threads.@threads for k = 1:nbl
        zXbl[k] = cscale(Xbl[k], object.xmeans[k], object.xscales[k])
    end
    zXbl = blockscal(zXbl, object.bscales).Xbl
    @inbounds for k = 1:nbl
        zXbl[k] .= sqrtw .* zXbl[k]
    end
    X = reduce(hcat, zXbl)
    # Explained_X
    ssk = zeros(nbl)
    @inbounds for k = 1:nbl
        ssk[k] = ssq(zXbl[k])
    end
    sstot = sum(ssk)
    tt = object.TT
    tt_adj = colsum(object.P.^2) .* tt
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, 
        pvar = pvar, cumpvar = cumpvar)     
    # Correlation between the original X-variables
    # and the global scores
    z = cor(X, sqrtw .* object.T)  
    corx2t = DataFrame(z, string.("lv", 1:nlv))
    # Correlation between the X-block scores and the global scores 
    z = list(nlv, Matrix)
    @inbounds for a = 1:nlv
        z[a] = cor(object.Tb[a], sqrtw .* object.T[:, a])
    end
    cortb2t = DataFrame(reduce(hcat, z), string.("lv", 1:nlv))
    # Redundancies (Average correlations) Rd(X, t) 
    # between each X-block and each global score
    z = list(nbl, Matrix)
    @inbounds for k = 1:nbl
        z[k] = rd(zXbl[k], sqrtw .* object.T)
    end
    rdx = DataFrame(reduce(vcat, z), string.("lv", 1:nlv))         
    # Specific weights of each block on each X-global score
    sal2 = nothing
    if !isnothing(object.lb)
        lb2 = colsum(object.lb.^2)
        sal2 = scale(object.lb.^2, lb2)
        sal2 = DataFrame(sal2, string.("lv", 1:nlv))
    end
    # Output
    (explvarx = explvarx, corx2t, cortb2t, rdx, sal2)
end
