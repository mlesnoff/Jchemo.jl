
"""
    eposvd(D; nlv)
Calibration transfer of spectral data by orthogonalization
* `D` : Data (m, p) containing the "detrimental" information on which the spectra
    have to be orthogonalized.
* `nlv` : Nb. of first loadings vectors of D considered for the orthogonalization.

The objective is to remove from a dataset X (n, p) some "detrimental" 
information (e.g. humidity patterns in signals, multiple spectrometers, etc.) 
defined by a dataset `D` (m, p). The method orthogonalizes the observations 
(rows of X) to the detrimental sub-space defined by the first `nlv` 
loadings vectors computed from a (non-centered) PCA of `D`.

Function `eposvd` makes a SVD factorization of `D` and returns 
two matrices:
* `M` (p, p) : The orthogonalization matrix that can be used to correct the X-data.
* `P` (p, `nlv`) : The matrix of the loading vectors of D. 

Any dataset X can be corrected from the detrimental information `D` 
by computing X_corrected = X * `M`.

Matrix `D` can be built from different choices. Two common methods are:
* EPO (Roger et al. 2003, 2018): `D` is built from differences between spectra
    collected from the different conditions. 
* TOP (Andrew & Fearn 2004): Each row of `D` is the mean spectrum for an instrument.

# References
Andrew, A., Fearn, T., 2004. Transfer by orthogonal projection: making near-infrared 
calibrations robust to between-instrument variation. Chemometrics and Intelligent 
Laboratory Systems 72, 51–56. https://doi.org/10.1016/j.chemolab.2004.02.004

Roger, J.-M., Chauchard, F., Bellon-Maurel, V., 2003. EPO-PLS external parameter 
orthogonalisation of PLS application to temperature-independent measurement 
of sugar content of intact fruits. 
Chemometrics and Intelligent Laboratory Systems 66, 191-204. 
https://doi.org/10.1016/S0169-7439(03)00051-0

Roger, J.-M., Boulet, J.-C., 2018. A review of orthogonal projections for calibration. 
Journal of Chemometrics 32, e3045. https://doi.org/10.1002/cem.3045

Zeaiter, M., Roger, J.M., Bellon-Maurel, V., 2006. Dynamic orthogonal projection. 
A new method to maintain the on-line robustness of multivariate calibrations. 
Application to NIR-based monitoring of wine fermentations. Chemometrics and Intelligent 
Laboratory Systems, 80, 227–235. https://doi.org/10.1016/j.chemolab.2005.06.011

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "caltransfer.jld2") 
@load db dat
pnames(dat)
X1cal = dat.X1cal
X2cal = dat.X2cal
X1val = dat.X1val
X2val = dat.X2val

D = X1cal - X2cal
nlv = 2
res = eposvd(D; nlv = nlv)
res.M      # orthogonalization matrix
res.P      # detrimental directions (columns of P = loadings of D)

# Corrected matrices

zX1 = X1val * res.M    
zX2 = X2val * res.M    

i = 1
f, ax = lines(zX1[i, :])
lines!(ax, zX2[i, :])
f
```
""" 
function eposvd(D; nlv)
    D = ensure_mat(D)
    m, p = size(D)
    nlv = min(nlv, m, p)
    Id = Diagonal(I, p)
    if nlv == 0 
        M = Id
        P = nothing
    else 
        P = svd(D).V[:, 1:nlv]
        M = Id - P * P'
    end
    (M = M, P = P)
end



