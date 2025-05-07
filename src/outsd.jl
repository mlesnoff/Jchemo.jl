"""
    outsd(fitm)
Compute an outlierness from PCA/PLS score distance (SD).
* `fitm` : The preliminary model (e.g. object `fitm` returned by function `pcasvd`) that was fitted on 
    the data.

In this method, the outlierness `d` of an observation is defined by its score distance (SD), ie. the Mahalanobis 
distance between the projection of the observation on the score plan defined by the fitted (e.g. PCA) model and the 
"center" (always defined by zero) of the score plan.

## References
M. Hubert, V. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust principal components 
analysis. Technometrics, 47, 64-79.

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
## Six of the samples (25, 26, and 36-39) contain added alcohol.
s = [25; 26; 36:39]
typ = zeros(Int, n)
typ[s] .= 1
#plotsp(X, wl; xlabel = "Wavelength (nm)", ylabel = "Absorbance").f

model = pcaout(; nlv = 3)
fit!(model, X) 
fitm = model.fitm ;
res = outsd(fitm) ;
@names res
f, ax = plotxy(1:n, res.d, typ, xlabel = "Obs. index", ylabel = "Outlierness")
text!(ax, 1:n, res.d; text = string.(1:n), fontsize = 10)
f
```
"""
function outsd(fitm)
    Q = eltype(fitm.T)
    nlv = nco(fitm.T)
    tscales = colstd(fitm.T, fitm.weights)
    T = fscale(fitm.T, tscales)
    d2 = vec(euclsq(T, zeros(Q, nlv)'))   # the center is defined as 0
    d = sqrt.(d2)
    (d = d,)
end
