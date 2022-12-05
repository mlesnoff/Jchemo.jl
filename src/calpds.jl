struct CalTransfPds
    fm
    s
end

"""
    calpds(X1, X2; fun = mlrpinv, m = 5, kwargs...)
Calibration transfer of spectral data with piecewise direct standardization (PDS).
* `X1` : Target (standart) X-data (n, p).
* `X2` : X-data (n, p) to transfer to the standart.
* `fun` : Function used for fitting the transfer model.  
* `m` : Half-window size (nb. points left/right to the target wavelength) 
* `kwargs` : Optional arguments for `fun`.

To predict wavelength i in `X1`, the window in `X2` is :

i - m, i - m + 1, ..., i, ..., i + m - 1, i + m

`X1` and `X2` are assumed to represent the same n samples. 

## References
Bouveresse, E., Massart, D.L., 1996. Improvement of the piecewise direct standardisation procedure 
for the transfer of NIR spectra for multivariate calibration. Chemometrics and Intelligent Laboratory 
Systems 32, 201–213. https://doi.org/10.1016/0169-7439(95)00074-7

Y. Wang, D. J. Veltkamp, and B. R. Kowalski, “Multivariate Instrument Standardization,” 
Anal. Chem., vol. 63, no. 23, pp. 2750–2756, 1991, doi: 10.1021/ac00023a016.

Wülfert, F., Kok, W.Th., Noord, O.E. de, Smilde, A.K., 2000. Correction of Temperature-Induced 
Spectral Variation by Continuous Piecewise Direct Standardization. Anal. Chem. 72, 1639–1644.
https://doi.org/10.1021/ac9906835

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
n = nro(X1cal)
m = nro(X1val)

fm = calpds(X1cal, X2cal; fun = plskern, nlv = 1, m = 2) ;
pred = Jchemo.predict(fm, X2val).pred
i = 1
f, ax = lines(X1val[i, :])
lines!(X2val[i, :])
lines!(pred[i, :], linestyle = "--")
f
```
""" 
function calpds(X1, X2; fun = mlrpinv, m = 5, kwargs...)
    p = nco(X1)
    fm = list(p)
    s = list(p)
    zm = repeat([m], p)
    zm[1:m] .= collect(1:m) .- 1
    zm[(p - m + 1):p] .= collect(m:-1:1) .- 1
    @inbounds for i = 1:p
        s[i] = collect((i - zm[i]):(i + zm[i]))
        fm[i] = fun(X2[:, s[i]], X1[:, i]; kwargs...)
    end
    CalTransfPds(fm, s)
end

"""
    predict(object::CalTransfPds, X; kwargs...)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `kwargs` : Optional arguments.
""" 
function predict(object::CalTransfPds, X; kwargs...)
    X = ensure_mat(X)
    m, p = size(X)
    pred = similar(X, m, p)
    for i = 1:p 
        pred[:, i] .= predict(object.fm[i], X[:, object.s[i]]; kwargs...).pred
    end
    (pred = pred,)
end



