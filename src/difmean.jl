"""
    difmean(X1, X2; normx::Bool = false)
Compute a 1-D detrimental matrix by difference of the column-means of two X-datas.
* `X1` : Spectra (n1, p).
* `X2` : Spectra (n2, p).
Keyword arguments:
* `normx` : Boolean. If `true`, the column-means vectors 
    of `X1` and `X2` are normed before computing their difference.

The function returns a matrix `D` (1, p) computed by the difference 
between two mean-spectra, i.e. the column-means of `X1` and `X2`. 

`D` is assumed to contain the detrimental information that can 
be removed (by orthogonalization) from `X1` and `X2`  for 
calibration transfer. For instance, `D` can be used as input of 
function `eposvd`. 

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/caltransfer.jld2")
@load db dat
pnames(dat)
X1cal = dat.X1cal
X1val = dat.X1val
X2cal = dat.X2cal
X2val = dat.X2val

## The objective is to remove a detrimental 
## information (here, D) from spaces X1 and X2
D = difmean(X1cal, X2cal).D
res = eposvd(D; nlv = 1)
## Corrected Val matrices
X1val_c = X1val * res.M
X2val_c = X2val * res.M

i = 1
f = Figure(size = (800, 300))
ax1 = Axis(f[1, 1])
ax2 = Axis(f[1, 2])
lines!(ax1, X1val[i, :]; label = "x1")
lines!(ax1, X2val[i, :]; label = "x2")
axislegend(ax1, position = :cb, framevisible = false)
lines!(ax2, X1val_c[i, :]; label = "x1_correct")
lines!(ax2, X2val_c[i, :]; label = "x2_correct")
axislegend(ax2, position = :cb, framevisible = false)
f
```
"""
function difmean(X1, X2; normx::Bool = false)
    xmeans1 = colmean(X1)
    xmeans2 = colmean(X2)
    if normx
        xmeans1 ./= norm(xmeans1)
        xmeans2 ./= norm(xmeans2)
    end
    D = (xmeans1 - xmeans2)'
    (D = D, xmeans1, xmeans2)
end
