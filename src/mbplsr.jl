"""
    mbplsr(; kwargs...)
    mbplsr(Xbl, Y; kwargs...)
    mbplsr(Xbl, Y, weights::Weight; kwargs...)
    mbplsr!(Xbl::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Multiblock PLSR (MBPLSR).
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` from (n, p) data.  
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. global latent variables (LVs; = scores) to compute.
* `bscal` : Type of block scaling. See function `blockscal` for possible values.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` and `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

This function runs a PLSR on {X, `Y`} where X is the horizontal concatenation of the blocks in `Xbl`. The function 
    gives the same global LVs and predictions as function `mbplswest`, but is much faster.

Function `summary` returns: 
* `explvarx` : Proportion of the total X inertia (squared Frobenious norm) explained by the global LVs.
* `rvxbl2t` : RV coefficients between each block and the global LVs.
* `rdxbl2t` : Rd coefficients between each block (= Xbl[k]) and the global LVs.
* `corx2t` : Correlation between the X-variables and the global LVs.  

## Examples
```julia
using Jchemo, JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "ham.jld2") 
@load db dat
@names dat 
X = dat.X
Y = dat.Y
y = Y.c1
group = dat.group
listbl = [1:11, 12:19, 20:25]
s = 1:6
Xbltrain = mblock(X[s, :], listbl)
ytrain = y[s]
Xbltest = mblock(rmrow(X, s), listbl)
ytest = rmrow(y, s) 
ntrain = nro(ytrain) 
ntest = nro(ytest) 
ntot = ntrain + ntest 
(ntot = ntot, ntrain , ntest)

nlv = 3
bscal = :frob
model = mbplsr(; nlv, bscal)
fit!(model, Xbltrain, ytrain)
@names model 
@names model.fitm
@head model.fitm.T
@head transf(model, Xbltrain)
transf(model, Xbltest)

res = predict(model, Xbltest)
res.pred 
rmsep(res.pred, ytest)

res = summary(model, Xbltrain) ;
@names res 
res.explvarx
res.rvxbl2t
res.rdxbl2t
res.cortbl2t
res.corx2t 

## This MBPLSR can also be implemented with function pip

model1 = blockscal(; bscal, centr = true) ;
model2 = mbconcat()
model3 = plskern(; nlv, scal = false) ;
model = pip(model1, model2, model3)
fit!(model, Xbltrain, ytrain)
@head T =  model.model[3].fitm.T  # = transf(model, Xbltrain)
transf(model, Xbltest)
predict(model, Xbltest).pred 
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
Compute latent variables (LVs; = scores) from a fitted model.
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
    ## Block scaling
    zXbl = transf(object.fitmbl, Xbl)
    X = fconcat(zXbl)
    ## Proportion of the total X-inertia explained by each global LV
    ssk = zeros(Q, nbl)
    @inbounds for k in eachindex(Xbl)
        ssk[k] = frob2(zXbl[k], object.weights)
    end
    tt = object.fitm.TT
    tt_adj = (colnorm(object.fitm.V).^2) .* tt  # tt_adj[a] = p[a]'p[a] * tt[a]
    pvar = tt_adj / sum(ssk)
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## RV between each Xk and the global LVs
    nam = string.("lv", 1:nlv)
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
    ## Correlation between the X-variables and the global LVs 
    z = corm(X, object.T, object.weights)  
    corx2t = DataFrame(z, nam)      
    (explvarx = explvarx, rvxbl2t, rdxbl2t, corx2t)
end


