"""
    difmean(X1, X2; normx = false)
Compute a 1-D detrimental matrix (for calibration transfer) by difference of 
    the column means.
* `X1` : Matrix of spectra (n1, p).
* `X2` : Matrix of spectra (n2, p).
* `normx` : Boolean. If `true`, the column means vectors 
    of `X1` and `X2` are normed before computing their difference.
    Default is `false`.

The function returns a matrix D (1, p) containing the detrimental information
that has to be removed from spectra `X1` and `X2` for calibration transfer 
by orthogonalization (e.g. input for function `eposvd`). 

Matrix D is computed by the difference between the two mean spectra 
(column means of `X1` and `X2`).

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/caltransfer.jld2") 
@load db dat
pnames(dat)
X1cal = dat.X1cal
X2cal = dat.X2cal
X1val = dat.X1val
X2val = dat.X2val

D = difmean(X1cal, X2cal).D 
res = eposvd(D; nlv = 1)
## Corrected matrices
X1 = X1val * res.M    
X2 = X2val * res.M    

i = 1
f = Figure(resolution = (500, 300))
ax = Axis(f[1, 1])
lines!(X1[i, :]; label = "x1_correct")
lines!(ax, X2[i, :]; label = "x2_correct")
axislegend(position = :rb, framevisible = false)
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
