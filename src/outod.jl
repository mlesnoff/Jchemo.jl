"""
    outod(fitm, X)
Compute outlierness from PCA/PLS orthogonal distance (OD).
* `fitm` : The reduction dimension model that was fitted on the data (e.g., object `fitm` returned by functions 
    `pcasvd` or `plskern`).
* `X` : X-data (n, p) on which was fitted model `fitm`.

In this function, outlierness `d` of an observation is defined by its orthogonal distance (= 'X-residuals'), 
ie. the Euclidean distance between the observation and its projection to the score plan fitted by the model
(e.g., PCA or PLS).

## References
M. Hubert, V. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust principal components analysis. 
Technometrics, 47, 64-79.

K. Vanden Branden, M. Hubert (2005). Robust classification in high dimension based on the SIMCA method. 
Chem. Lab. Int. Syst, 79, 10-21.

K. Varmuza, V. Filzmoser (2009). Introduction to multivariate statistical analysis in chemometrics. 
CRC Press, Boca Raton.

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
res = outod(fitm, X) ;
#res = outsdod(fitm, X)
@names res
f, ax = plotxy(1:n, res.d, typ; xlabel = "Observation index", ylabel = "Outlierness")
text!(ax, 1:n, res.d; text = string.(1:n), fontsize = 10)
f
```
"""
function outod(fitm, X)
    E = xresid(fitm, X)
    d = rownorm(E)
    (d = d,)
end
