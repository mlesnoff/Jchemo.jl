struct CalDs
    fm
end


"""
    calds(Xst, X; fun = mlrpinv, kwargs...)
Direct standardization (PDS) for calibration transfer of spectral data.
* `Xst` : Standart spectra, (n, p).
* `X` : Spectra to transfer to the standart, (n, p).
* `fun` : Function used for fitting the transfer model.  
* `kwargs` : Optional arguments for `fun`.

`Xst` and `X` must represent the same n samples.

The objective is to transform spectra `X` to spectra as close 
as possible as the standard `Xst`. The principle of the method is to fit models 
predicting `Xst` from `X.

## References

Y. Wang, D. J. Veltkamp, and B. R. Kowalski, “Multivariate Instrument Standardization,” 
Anal. Chem., vol. 63, no. 23, pp. 2750–2756, 1991, doi: 10.1021/ac00023a016.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "caltransfer.jld2") 
@load db dat
pnames(dat)

## Standart
Xstcal = dat.X1cal
Xstval = dat.X1val
## To be transfered
Xcal = dat.X2cal
Xval = dat.X2val

n = nro(Xstcal)
m = nro(Xstval)

fm = calds(Xstcal, Xcal; fun = mlrpinv) ;
#fm = calds(Xstcal, Xcal; fun = pcr, nlv = 15) ;
#fm = calds(Xstcal, Xcal; fun = plskern, nlv = 15) ;
pred = Jchemo.predict(fm, Xval).pred
i = 1
f = Figure(resolution = (500, 300))
ax = Axis(f[1, 1])
lines!(Xstval[i, :]; label = "xst")
lines!(ax, Xval[i, :]; label = "x")
lines!(pred[i, :]; linestyle = "--", label = "x_transfered")
axislegend(position = :rb)
f
```
""" 
function calds(Xst, X; fun = mlrpinv, kwargs...)
    fm = fun(X, Xst; kwargs...)
    CalDs(fm)
end

"""
    predict(object::CalDs, X; kwargs...)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `kwargs` : Optional arguments.
""" 
function predict(object::CalDs, X; kwargs...)
    predict(object.fm, X; kwargs...)
end


