struct Mbplsr
    fm
    T::Matrix{Float64}
    R::Matrix{Float64}
    C::Matrix{Float64}
    bscales::Vector{Float64}
    xmeans::Vector{Vector{Float64}}
    xscales::Vector{Vector{Float64}}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end

"""
    mbplsr(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "none", scal = false)
    mbplsr!(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "none", scal = false)
Multiblock PLSR (MBPLSR) - Fast version.
* `Xbl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `Y` : Y-data.
* `weights` : Weights of the observations (rows). 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscal` : Type of block scaling. 
    Possible values are: "none", "frob", "mfa", "ncol", "sd". 
    See functions `blockscal`.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` and 
    of `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

This is the PLSR (X, `Y`) where X is the horizontal concatenation of the blocks in `Xbl`.
The function gives the same results as function `mbplswest`.

## Examples
```julia
using JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "ham.jld2") 
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

bscal = "none"
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

function mbplsr(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "none", scal = false)
    nbl = length(Xbl)  
    zXbl = list(nbl, Matrix{Float64})
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    mbplsr!(zXbl, copy(ensure_mat(Y)), weights; nlv = nlv, 
        bscal = bscal, scal = scal)
end

function mbplsr!(Xbl, Y, weights = ones(nro(Xbl[1])); nlv, 
        bscal = "none", scal = false)
    nbl = length(Xbl)
    Y = ensure_mat(Y)
    q = nco(Y)
    weights = mweight(weights)
    xmeans = list(nbl, Vector{Float64})
    xscales = list(nbl, Vector{Float64})
    Threads.@threads for k = 1:nbl
        xmeans[k] = colmean(Xbl[k], weights) 
        xscales[k] = ones(nco(Xbl[k]))
        if scal 
            xscales[k] = colstd(Xbl[k], weights)
            Xbl[k] .= cscale(Xbl[k], xmeans[k], xscales[k])
        else
            Xbl[k] .= center(Xbl[k], xmeans[k])
        end
    end
    ymeans = colmean(Y, weights)
    yscales = ones(q)
    if scal 
        yscales .= colstd(Y, weights)
        cscale!(Y, ymeans, yscales)
    else
        center!(Y, ymeans)
    end
    if bscal == "none" 
        bscales = ones(nbl)
    else
        bscal == "frob" ? res = blockscal_frob(Xbl, weights) : nothing
        bscal == "mfa" ? res = blockscal_mfa(Xbl, weights) : nothing
        bscal == "ncol" ? res = blockscal_ncol(Xbl) : nothing
        bscal == "sd" ? res = blockscal_sd(Xbl, weights) : nothing
        Xbl = res.X
        bscales = res.bscales
    end
    X = reduce(hcat, Xbl)
    fm = plskern(X, Y, weights; nlv = nlv, scal = false)
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
    n, nlv = size(object.T)
    nbl = length(Xbl)
    sqrtw = sqrt.(object.weights)
    zXbl = list(nbl, Matrix{Float64})
    Threads.@threads for k = 1:nbl
        zXbl[k] = cscale(Xbl[k], object.xmeans[k], object.xscales[k])
    end
    zXbl = blockscal(zXbl, object.bscales).X
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
    tt = object.fm.TT
    tt_adj = vec(sum(object.fm.P.^2, dims = 1)) .* tt
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    # Correlation between the global scores and the original variables 
    z = cor(X, object.T)  
    cort2x = DataFrame(z, string.("lv", 1:nlv))
    ## Redundancies (Average correlations) Rd(X, tx) and Rd(Y, ty) between each block and each global score
    z = list(nbl, Matrix{Float64})
    @inbounds for k = 1:nbl
        z[k] = rd(zXbl[k], object.T)
    end
    rdx = DataFrame(reduce(vcat, z), string.("lv", 1:nlv))       
    # Outputs
    (explvarx = explvarx, cort2x, rdx)
end
