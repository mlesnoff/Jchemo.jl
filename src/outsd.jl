"""
    outsd(fitm)
Compute outlierness from PCA/PLS score distance (SD).
* `fitm` : The reduction dimension model that was fitted on the data (e.g., object `fitm` returned by functions 
    `pcasvd` or `plskern`).

In this function, outlierness `d` of an observation is defined by its score distance (SD), ie. the Mahalanobis 
distance between the projection of the observation on the score plan fitted by the model (e.g., PCA or PLS) and the 
'center' of the score plan (in this function always defined by zero).

## References
M. Hubert, V. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust principal components analysis. 
Technometrics, 47, 64-79.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "octane.jld2")
@load db dat
X = dat.X
wlst = names(X)
wl = parse.(Float64, wlst)
n, p = size(X)
## Six of the samples (25, 26, and 36-39) contain added alcohol
s = [25; 26; 36:39]
typ = zeros(Int, n)
typ[s] .= 1
#plotsp(X, wl; xlabel = "Wavelength (nm)", ylabel = "Absorbance").f

model = pcaout(; nlv = 3)
fit!(model, X) 
fitm = model.fitm ;
res = outsd(fitm) ;
@names res
f, ax = plotxy(1:n, res.d, typ, xlabel = "Observation index", ylabel = "Outlierness")
text!(ax, 1:n, res.d; text = string.(1:n), fontsize = 10)
f
```
"""
function outsd(fitm)
    T = copy(fitm.T)
    Q = eltype(T)
    nlv = nco(T)
    tscales = colstd(T, fitm.weights)
    fscale!(T, tscales)
    centr = zeros(Q, nlv)     # the center is defined as 0
    d2 = vec(eucl2(T, centr'))   
    (d = sqrt.(d2), tscales)
end
