struct CalDs
    fm
end


"""
    calds(Xt, X; fun = mlrpinv, kwargs...)
Direct targetization (PDS) for calibration transfer of spectral data.
* `Xt` : Target spectra, (n, p).
* `X` : Spectra to transfer to the target, (n, p).
* `fun` : Function used for fitting the transfer model.  
* `kwargs` : Optional arguments for `fun`.

`Xt` and `X` must represent the same n standard samples.

The objective is to transform spectra `X` to spectra as close 
as possible as the target `Xt`. The principle of the method is to fit models 
predicting `Xt` from `X.

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

## Target
Xtcal = dat.X1cal
Xtval = dat.X1val
## To predict
Xcal = dat.X2cal
Xval = dat.X2val

n = nro(Xtcal)
m = nro(Xtval)

fm = calds(Xtcal, Xcal; fun = mlrpinv) ;
#fm = calds(Xtcal, Xcal; fun = pcr, nlv = 15) ;
#fm = calds(Xtcal, Xcal; fun = plskern, nlv = 15) ;
pred = Jchemo.predict(fm, Xval).pred     # Transfered spectra

i = 1
f = Figure(resolution = (500, 300))
ax = Axis(f[1, 1])
lines!(Xtval[i, :]; label = "xt")
lines!(ax, Xval[i, :]; label = "x")
lines!(pred[i, :]; linestyle = "--", label = "x_transf")
axislegend(position = :rb, framevisible = false)
f
```
""" 
function calds(Xt, X; fun = mlrpinv, kwargs...)
    fm = fun(X, Xt; kwargs...)
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


