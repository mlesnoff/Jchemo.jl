"""
    mbplsr(; kwargs...)
    mbplsr(Xbl, Y; kwargs...)
    mbplsr(Xbl, Y, weights::Weight; kwargs...)
    mbplsr!(Xbl::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Multiblock PLSR (MBPLSR).
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. global latent variables (LVs = scores) to compute.
* `bscal` : Type of block scaling. See function `blockscal`
    for possible values.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    and `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

This function runs a PLSR on {X, `Y`} where X is the horizontal 
concatenation of the blocks in `Xbl`. The function gives the 
same global LVs and predictions as function `mbplswest`, but is much faster.

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
model = mbplsr(; nlv, bscal, scal)
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
res.rdx
```
"""
mbplsr(; kwargs...) = JchemoModel(mbplsr, nothing, kwargs)

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
    @inbounds for k in eachindex(Xbl)
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    mbplsr!(zXbl, copy(ensure_mat(Y)), weights; kwargs...)
end

function mbplsr!(Xbl::Vector, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
    par = recovkw(ParMbplsr, kwargs).par
    Q = eltype(Xbl[1][1, 1])
    isa(Y, BitMatrix) ? Y = convert.(Q, Y) : nothing
    q = nco(Y)
    fitmbl = blockscal(Xbl, weights; centr = true, scal = par.scal, bscal = par.bscal)
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
    fitm = plskern(X, Y, weights; nlv = par.nlv, scal = false)
    Mbplsr(fitm, fitm.T, fitm.R, fitm.C, fitmbl, ymeans, yscales, weights, par)
end

""" 
    transf(object::Union{Mbplsr, Mbplswest}, Xbl; nlv = nothing)
Compute latent variables (LVs = scores) from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transf(object::Union{Mbplsr, Mbplswest}, Xbl; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    zXbl = transf(object.fitmbl, Xbl)    
    fconcat(zXbl) * vcol(object.R, 1:nlv) 
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
    isnothing(nlv) ? nlv = a : nlv = (min(a, minimum(nlv)):min(a, maximum(nlv)))
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
    zXbl = transf(object.fitmbl, Xbl)
    ## Metric
    sqrtw = sqrt.(object.weights.w)
    @inbounds for k in eachindex(Xbl)
        #fweight!(zXbl[k], sqrtw)
    end
    X = fconcat(zXbl)
    ## Proportion of the X-inertia explained per global LV
    ssk = zeros(Q, nbl)
    @inbounds for k in eachindex(Xbl)
        #ssk[k] = frob2(zXbl[k])
        ssk[k] = frob2(zXbl[k], object.weights)
    end
    tt = object.fitm.TT
    tt_adj = (colnorm(object.fitm.V).^2) .* tt  # tt_adj[a] = p[a]'p[a] * tt[a]
    pvar = tt_adj / sum(ssk)
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## Correlation between the original X-variables and the global LVs 
    #z = cor(X, fweight(object.T, sqrtw))
    z = corm(X, object.T, object.weights)  
    corx2t = DataFrame(z, string.("lv", 1:nlv))
    ## Redundancies (Average correlations) Rd(X, t) 
    ## between each X-block and each global score
    z = list(Matrix{Q}, nbl)
    @inbounds for k in eachindex(Xbl)
        z[k] = rd(zXbl[k], fweight(object.T, sqrtw))
    end
    rdx = DataFrame(reduce(vcat, z), string.("lv", 1:nlv))       
    ## Outputs
    (explvarx = explvarx, corx2t, rdx)
end
