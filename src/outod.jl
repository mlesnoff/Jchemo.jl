"""
    outod(fitm, X)
Compute an outlierness from PCA/PLS orthogonal distance (OD).
* `fitm` : The preliminary model (e.g. object `fitm` returned by function `pcasvd`) that was fitted on 
    the data.
* `X` : Training X-data (n, p), on which was fitted the model `fitm`.

In this method, the outlierness `d` of an observation is the orthogonal distance (=  'X-residuals') of this 
observation, ie. the Euclidean distance between the observation and its projection on the score plan defined by 
the fitted (e.g. PCA) model (e.g. Hubert et al. 2005, Van Branden & Hubert 2005 p. 66, Varmuza & Filzmoser 
2009 p. 79).

## References
M. Hubert, V. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust principal components 
analysis. Technometrics, 47, 64-79.

K. Vanden Branden, M. Hubert (2005). Robuts classification in high dimension based on the SIMCA method. 
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
function outod(fitm, X)
    E = xresid(fitm, X)
    d = rownorm(E)
    (d = d,)
end
