"""
    mbplsr(; kwargs...)
    mbplsr(Xbl, Y; kwargs...)
    mbplsr(Xbl, Y, weights::Weight; kwargs...)
    mbplsr!(Xbl::Matrix, Y::Matrix, weights::Weight; kwargs...)
Multiblock PLSR (MBPLSR).
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
    * `nlv` : Nb. latent variables (LVs = scores T) to compute.
    * `bscal` : Type of block scaling. Possible values are:
        `:none`, `:frob`. See functions `blockscal`.
    * `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
        and `Y` is scaled by its uncorrected standard deviation 
        (before the block scaling).

This function runs a PLSR on {X, `Y`} where X is the horizontal 
concatenation of the blocks in `Xbl`. The function gives the 
same results as function `mbplswest`, but is much faster.

## Examples
```julia
using JchemoData, JLD2
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
Xbl_train = mblock(X[s, :], listbl)
Xbl_test = mblock(rmrow(X, s), listbl)
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
mod = mbplsr(; nlv, 
    bscal, scal)
fit!(mod, Xbl_train, ytrain)
pnames(mod) 
pnames(mod.fm)
@head mod.fm.T
@head transf(mod, Xbl_train)
transf(mod, Xbl_test)

res = predict(mod, Xbl_test)
res.pred 
rmsep(res.pred, ytest)

res = summary(mod, Xbl_train) ;
pnames(res) 
res.explvarx
res.corx2t 
res.rdx
```
"""
function mbplsr(Xbl, Y; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    weights = mweight(ones(Q, n))
    mbplsr(Xbl, Y, weights; kwargs...)
end

function mbplsr(Xbl, Y, weights::Weight; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)  
    zXbl = list(Matrix{Q}, nbl)
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    mbplsr!(zXbl, copy(ensure_mat(Y)), weights; kwargs...)
end

function mbplsr!(Xbl::Vector, Y::Matrix, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert in([:none, :frob])(par.bscal) "Wrong value for argument 'bscal'."
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)
    q = nco(Y)
    xmeans = list(Vector{Q}, nbl)
    xscales = list(Vector{Q}, nbl)
    Threads.@threads for k = 1:nbl
        xmeans[k] = colmean(Xbl[k], weights) 
        xscales[k] = ones(Q, nco(Xbl[k]))
        if par.scal 
            xscales[k] = colstd(Xbl[k], weights)
            fcscale!(Xbl[k], xmeans[k], xscales[k])
        else
            fcenter!(Xbl[k], xmeans[k])
        end
    end
    ymeans = colmean(Y, weights)
    yscales = ones(Q, q)
    if par.scal 
        yscales .= colstd(Y, weights)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(Y, ymeans)
    end
    par.bscal == :none ? bscales = ones(nbl) : nothing
    if par.bscal == :frob
        res = fblockscal_frob(Xbl, weights) 
        bscales = res.bscales
        Xbl = res.Xbl
    end
    X = reduce(hcat, Xbl)
    fm = plskern(X, Y, weights; nlv = par.nlv, scal = false)
    Mbplsr(fm, fm.T, fm.R, fm.C, 
        bscales, xmeans, xscales, ymeans, yscales, 
        weights, kwargs, par)
end

""" 
    transf(object::Union{Mbplsr, Mbplswest}, Xbl; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transf(object::Union{Mbplsr, Mbplswest}, Xbl; nlv = nothing)
    Q = eltype(Xbl[1][1, 1])
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    nbl = length(Xbl)
    zXbl = list(Matrix{Q}, nbl)
    Threads.@threads for k = 1:nbl
        zXbl[k] = fcscale(Xbl[k], object.xmeans[k], object.xscales[k])
    end
    res = fblockscal(zXbl, object.bscales)
    reduce(hcat, res.Xbl) * vcol(object.R, 1:nlv) 
end

"""
    predict(object::Union{Mbplsr, Mbplswest}, Xbl; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Union{Mbplsr, Mbplswest}, Xbl; nlv = nothing)
    Q = eltype(Xbl[1][1, 1])
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = (max(0, minimum(nlv)):min(a, maximum(nlv)))
    le_nlv = length(nlv)
    T = transf(object, Xbl)
    pred = list(Matrix{Q}, le_nlv)
    @inbounds  for i = 1:le_nlv
        znlv = nlv[i]
        W = Diagonal(object.yscales)
        beta = object.C[:, 1:znlv]'
        int = object.ymeans'
        pred[i] = int .+ vcol(T, 1:znlv) * beta * W 
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end

"""
    summary(object::Mbplsr, Xbl)
Summarize the fitted model.
* `object` : The fitted model.
* `Xbl` : The X-data that was used to 
    fit the model.
""" 
function Base.summary(object::Mbplsr, Xbl)
    Q = eltype(Xbl[1][1, 1])
    n, nlv = size(object.T)
    nbl = length(Xbl)
    sqrtw = sqrt.(object.weights.w)
    zXbl = list(Matrix{Q}, nbl)
    Threads.@threads for k = 1:nbl
        zXbl[k] = fcscale(Xbl[k], object.xmeans[k], object.xscales[k])
    end
    zXbl = fblockscal(zXbl, object.bscales).Xbl
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
    tt = object.fm.TT
    tt_adj = vec(sum(object.fm.P.^2, dims = 1)) .* tt
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, 
        pvar = pvar, cumpvar = cumpvar)
    ## Correlation between the original X-variables
    ## and the global scores 
    z = cor(X, sqrtw .* object.T)  
    corx2t = DataFrame(z, string.("lv", 1:nlv))
    ## Redundancies (Average correlations) Rd(X, t) 
    ## between each X-block and each global score
    z = list(Matrix{Q}, nbl)
    @inbounds for k = 1:nbl
        z[k] = rd(zXbl[k], sqrtw .* object.T)
    end
    rdx = DataFrame(reduce(vcat, z), 
        string.("lv", 1:nlv))       
    ## Outputs
    (explvarx = explvarx, corx2t, rdx)
end
