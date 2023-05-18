struct CalDs
    fm
end


"""
    calds(X, Xt; fun = mlrpinv, kwargs...)
Direct standardization (DS) for calibration transfer of spectral data.
* `X` : Spectra to transfer to the target (n, p).
* `Xt` : Target spectra (n, p).
* `fun` : Function used as transfer model.  
* `kwargs` : Optional arguments for `fun`.

`Xt` and `X` must represent the same n standard samples.

The objective is to transform spectra `X` to new spectra as close 
as possible as the target `Xt`. Method DS fits a model 
(defined in `fun`) that predicts `Xt` from `X`.

## References

Y. Wang, D. J. Veltkamp, and B. R. Kowalski, “Multivariate Instrument Standardization,” 
Anal. Chem., vol. 63, no. 23, pp. 2750–2756, 1991, doi: 10.1021/ac00023a016.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/caltransfer.jld2") 
@load db dat
pnames(dat)

## Target
Xtcal = dat.X1cal
Xtval = dat.X1val
## To transfer
Xcal = dat.X2cal
Xval = dat.X2val

n = nro(Xtcal)
m = nro(Xtval)

fm = calds(Xcal, Xtcal; fun = mlrpinv) ;
#fm = calds(Xcal, Xtcal; fun = pcr, nlv = 15) ;
#fm = calds(Xcal, Xtcal; fun = plskern, nlv = 15) ;
## Transfered spectra, expected to be close to Xtval
pred = Jchemo.predict(fm, Xval).pred 

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
function calds(X, Xt; fun = mlrpinv, kwargs...)
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


