"""
    soplsr(; kwargs...)
    soplsr(Xbl, Y; kwargs...)
    soplsr(Xbl, Y, weights::Weight; kwargs...)
    soplsr!(Xbl::Matrix, Y::Matrix, weights::Weight; kwargs...)
Multiblock sequentially orthogonalized PLSR (SO-PLSR).
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs = scores) to compute.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    and `Y` is scaled by its uncorrected standard deviation.

## References
Biancolillo et al. , 2015. Combining SO-PLS and linear 
discriminant analysis for multi-block classification. 
Chemometrics and Intelligent Laboratory Systems, 141, 58-67.

Biancolillo, A. 2016. Method development in the area of 
multi-block analysis focused on food analysis. PhD. 
University of copenhagen.

Menichelli et al., 2014. SO-PLS as an exploratory tool
for path modelling. Food Quality and Preference, 36, 122-134.

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

nlv = 2
#nlv = [2, 1, 2]
#nlv = [2, 0, 1]
scal = false
#scal = true
model = soplsr(; nlv, scal)
fit!(model, Xbltrain, ytrain)
pnames(model) 
pnames(model.fitm)
@head model.fitm.T
@head transf(model, Xbltrain)
transf(model, Xbltest)

res = predict(model, Xbltest)
res.pred 
rmsep(res.pred, ytest)
```
"""
soplsr(; kwargs...) = JchemoModel(soplsr, nothing, kwargs)

function soplsr(Xbl, Y; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    weights = mweight(ones(Q, n))
    soplsr(Xbl, Y, weights; kwargs...)
end

function soplsr(Xbl, Y, weights::Weight; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)  
    zXbl = list(Matrix{Q}, nbl)
    @inbounds for k in eachindex(Xbl)
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    soplsr!(zXbl, copy(ensure_mat(Y)), weights; kwargs...)
end

function soplsr!(Xbl::Vector, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParSoplsr, kwargs).par
    Q = eltype(Xbl[1][1, 1])
    Y = ensure_mat(Y)
    n = size(Xbl[1], 1)
    q = nco(Y)   
    nbl = length(Xbl)
    nlv = par.nlv
    length(nlv) == 1 ? nlv = repeat([nlv], nbl) : nothing  
    D = Diagonal(weights.w)
    ## 'bscal = :none' since block-scaling has no effect on SOPLS  
    fitmbl = blockscal(Xbl, weights; bscal = :none, centr = false, scal = par.scal)
    ## End
    transf!(fitmbl, Xbl)
    yscales = ones(Q, q)
    if par.scal 
        yscales .= colstd(Y, weights)
        fscale!(Y, yscales)
    end
    fitm = list(nbl)
    fit = similar(Xbl[1], n, q)
    b = list(nbl)
    ## Below, if 'scal' = true, 'fit' is in scale 'scaled-Y' 
    ## First block
    fitm[1] = plskern(Xbl[1], Y, weights; nlv = nlv[1], scal = false)  
    T = fitm[1].T
    fit .= predict(fitm[1], Xbl[1]).pred
    b[1] = nothing
    ## Other blocks
    if nbl > 1
        for i = 2:nbl
            b[i] = inv(T' * (D * T)) * T' * (D * Xbl[i])
            X = Xbl[i] - T * b[i]
            fitm[i] = plskern(X, Y - fit, weights; nlv = nlv[i], scal = false)  
            T = hcat(T, fitm[i].T)
            fit .+= predict(fitm[i], X).pred 
        end
    end
    Soplsr(fitm, T, fit, b, fitmbl, yscales, par)
end

""" 
    transf(object::Soplsr, Xbl)
Compute latent variables (LVs = scores) from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
""" 
function transf(object::Soplsr, Xbl)
    nbl = length(Xbl)
    zXbl = transf(object.fitmbl, Xbl)   
    T = transf(object.fitm[1], zXbl[1])
    if nbl > 1
        @inbounds for i = 2:nbl
            X = zXbl[i] - T * object.b[i]
            T = hcat(T, transf(object.fitm[i], X))
        end
    end
    T
end

"""
    predict(object::Soplsr, Xbl)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which predictions are computed.
""" 
function predict(object::Soplsr, Xbl)
    nbl = length(Xbl)
    zXbl = transf(object.fitmbl, Xbl)   
    T = transf(object.fitm[1], zXbl[1])
    pred =  object.fitm[1].ymeans' .+ T * object.fitm[1].C'
    if nbl > 1
        @inbounds for i = 2:nbl
            X = zXbl[i] - T * object.b[i]
            zT = transf(object.fitm[i], X)
            pred .+= object.fitm[i].ymeans' .+ zT * object.fitm[i].C'
            T = hcat(T, transf(object.fitm[i], X))
        end
    end
    pred .= pred .* object.yscales' 
    (pred = pred,)
end

