"""
    smbplsr(; kwargs...)
    smbplsr(Xbl, Y; kwargs...)
    smbplsr(Xbl, Y, weights::Weight; kwargs...)
    smbplsr!(Xbl::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Multiblock PLSR (MBPLSR).
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` from data (n, p).  
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. global latent variables (LVs; = scores) to compute.
* `bscal` : Type of block scaling. See function `blockscal` for possible values.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` and `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

Same as function `mbplsr` (see for details) except that a sparse PLSR (with function `splsr`) is run instead of a PLSR.

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
model = smbplsr(; nlv, bscal)
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
model3 = splsr(; nlv, scal = false) ;
model = pip(model1, model2, model3)
fit!(model, Xbltrain, ytrain)
@head T =  model.model[3].fitm.T  # = transf(model, Xbltrain)
transf(model, Xbltest)
predict(model, Xbltest).pred 
```
"""
smbplsr(; kwargs...) = JchemoModel(smbplsr, nothing, kwargs)

function smbplsr(Xbl, Y; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    weights = mweight(ones(Q, n))
    smbplsr(Xbl, Y, weights; kwargs...)
end

function smbplsr(Xbl, Y, weights::Weight; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)  
    zXbl = list(Matrix{Q}, nbl)
    @inbounds for k in eachindex(Xbl)
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    mbplsr!(zXbl, copy(ensure_mat(Y)), weights; kwargs...)
end

function smbplsr!(Xbl::Vector, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
    par = recovkw(ParSmbplsr, kwargs).par
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
    fitm = splsr(X, Y, weights; nlv = par.nlv, meth = par.meth, nvar = par.nvar, tol = par.tol,
        maxit = par.maxit, scal = false)
    Smbplsr(fitm, fitm.T, fitm.R, fitm.C, fitmbl, ymeans, yscales, weights, 
        fitm.sellv, fitm.sel, 
        par)
end
