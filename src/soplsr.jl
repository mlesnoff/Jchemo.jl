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
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling. See function `blockscal`
    for possible values.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    and `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

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
mod = soplsr(; nlv, scal)
fit!(mod, Xbltrain, ytrain)
pnames(mod) 
pnames(mod.fm)
@head mod.fm.T
@head transf(mod, Xbltrain)
transf(mod, Xbltest)

res = predict(mod, Xbltest)
res.pred 
rmsep(res.pred, ytest)
```
"""
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
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    soplsr!(zXbl, copy(ensure_mat(Y)), weights; kwargs...)
end

function soplsr!(Xbl::Vector, Y::Matrix, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs)
    Q = eltype(Xbl[1][1, 1])
    Y = ensure_mat(Y)
    n = size(Xbl[1], 1)
    q = nco(Y)   
    nbl = length(Xbl)
    nlv = par.nlv
    length(nlv) == 1 ? nlv = repeat([nlv], nbl) : nothing  
    D = Diagonal(weights.w)
    fmsc = blockscal(Xbl, weights; bscal = :none, centr = false, 
        scal = par.scal)
    transf!(fmsc, Xbl)
    yscales = ones(Q, q)
    if par.scal 
        yscales .= colstd(Y, weights)
        fscale!(Y, yscales)
    end
    fm = list(nbl)
    fit = similar(Xbl[1], n, q)
    b = list(nbl)
    ## Below, if 'scal' = true, 'fit' is in scale 'Y-scaled' 
    ## First block
    fm[1] = plskern(Xbl[1], Y, weights; nlv = nlv[1], scal = false)  
    T = fm[1].T
    fit .= predict(fm[1], Xbl[1]).pred
    b[1] = nothing
    ## Other blocks
    if nbl > 1
        for i = 2:nbl
            b[i] = inv(T' * (D * T)) * T' * (D * Xbl[i])
            X = Xbl[i] - T * b[i]
            fm[i] = plskern(X, Y - fit, weights; nlv = nlv[i], scal = false)  
            T = hcat(T, fm[i].T)
            fit .+= predict(fm[i], X).pred 
        end
    end
    Soplsr(fm, T, fit, b, fmsc, yscales, kwargs, par)
end

""" 
    transf(object::Soplsr, Xbl)
Compute latent variables (LVs = scores T) from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
""" 
function transf(object::Soplsr, Xbl)
    nbl = length(Xbl)
    zXbl = transf(object.fmsc, Xbl)   
    T = transf(object.fm[1], zXbl[1])
    if nbl > 1
        @inbounds for i = 2:nbl
            X = zXbl[i] - T * object.b[i]
            T = hcat(T, transf(object.fm[i], X))
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
    zXbl = transf(object.fmsc, Xbl)   
    T = transf(object.fm[1], zXbl[1])
    pred =  object.fm[1].ymeans' .+ T * object.fm[1].C'
    if nbl > 1
        @inbounds for i = 2:nbl
            X = zXbl[i] - T * object.b[i]
            zT = transf(object.fm[i], X)
            pred .+= object.fm[i].ymeans' .+ zT * object.fm[i].C'
            T = hcat(T, transf(object.fm[i], X))
        end
    end
    pred .= pred .* object.yscales' 
    (pred = pred,)
end

