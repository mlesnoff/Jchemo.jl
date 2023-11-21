"""
    mbplsr(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = :none, scal::Bool = false)
    mbplsr!(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = :none, scal::Bool = false)
Multiblock PLSR (MBPLSR) - Fast version.
* `Xbl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `Y` : Y-data.
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscal` : Type of `Xbl` block scaling (`:none`, `:frob`).
    See functions `blockscal`.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` and 
    of `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

PLSR (X, `Y`) where X is the horizontal concatenation of the blocks in `Xbl`.
The function gives the same results as function `mbplswest`, 
but is much faster.

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
fm = mbplsr(Xbl, y; nlv = nlv, bscal = bscal) ;
pnames(fm)
fm.T
Jchemo.transform(fm, Xbl_new)
[y Jchemo.predict(fm, Xbl).pred]
Jchemo.predict(fm, Xbl_new).pred

summary(fm, Xbl) 
```
"""
function mbplsr(Xbl, Y; par = Par())
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    weights = mweight(ones(Q, n))
    mbplsr(Xbl, Y, weights; par)
end

function mbplsr(Xbl, Y, weights::Weight; par = Par())
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)  
    zXbl = list(nbl, Matrix{Q})
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    mbplsr!(zXbl, copy(ensure_mat(Y)), 
        weights; par)
end

function mbplsr!(Xbl::Vector, Y::Matrix, weights::Weight; 
        par = Par())
    @assert in([:none, :frob])(par.bscal) "Wrong value for argument 'bscal'."
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)
    q = nco(Y)
    xmeans = list(nbl, Vector{Q})
    xscales = list(nbl, Vector{Q})
    Threads.@threads for k = 1:nbl
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
    par.bscal == :none ? bscales = ones(nbl) : nothing
    if par.bscal == :frob
        res = blockscal_frob(Xbl, weights) 
        bscales = res.bscales
        Xbl = res.Xbl
    end
    X = reduce(hcat, Xbl)
    fm = plskern(X, Y, weights; par)
    Mbplsr(fm, fm.T, fm.R, fm.C, 
        bscales, xmeans, xscales, ymeans, yscales, weights)
end

"""
    summary(object::Mbplsr, Xbl)
Summarize the fitted model.
* `object` : The fitted model.
* `Xbl` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Mbplsr, Xbl)
    Q = eltype(Xbl[1][1, 1])
    n, nlv = size(object.T)
    nbl = length(Xbl)
    sqrtw = sqrt.(object.weights.w)
    zXbl = list(nbl, Matrix{Q})
    Threads.@threads for k = 1:nbl
        zXbl[k] = cscale(Xbl[k], object.xmeans[k], object.xscales[k])
    end
    zXbl = blockscal(zXbl, object.bscales).Xbl
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
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, 
        cumpvar = cumpvar)
    # Correlation between the original X-variables
    # and the global scores 
    z = cor(X, sqrtw .* object.T)  
    corx2t = DataFrame(z, string.("lv", 1:nlv))
    # Redundancies (Average correlations) Rd(X, t) 
    # between each X-block and each global score
    z = list(nbl, Vector{Q})
    @inbounds for k = 1:nbl
        z[k] = rd(zXbl[k], sqrtw .* object.T)
    end
    rdx = DataFrame(reduce(vcat, z), string.("lv", 1:nlv))       
    # Outputs
    (explvarx = explvarx, corx2t, rdx)
end
