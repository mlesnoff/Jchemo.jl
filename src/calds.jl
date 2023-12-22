"""
    calds(X1, X2; fun = plskern, kwargs...)
Direct standardization (DS) for calibration transfer of spectral data.
* `X1` : Spectra (n, p) to transfer to the target.
* `X2` : Target spectra (n, p).
Keyword arguments:
* `fun` : Function used as transfer model.  
* `kwargs` : Optional arguments for `fun`.

`X1` and `X2` must represent the same n samples ("standards").

The objective is to transform spectra `X1` to new spectra as close 
as possible as the target `X2`. Method DS fits a model 
(defined in `fun`) that predicts `X2` from `X1`.

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
## Objects X1 and X2 are spectra collected 
## on the same samples. 
## X2 represents the target space. 
## We want to transfer X1 in the same space
## as X2.
## Data to transfer
X1cal = dat.X1cal
X1val = dat.X1val
n = nro(X1cal)
m = nro(X1val)
## Target space
X2cal = dat.X2cal
X2val = dat.X2val

## Fitting the model
fm = calds(X1cal, X2cal; 
    fun = plskern, nlv = 10) ;
#fm = calds(X1cal, X2cal; fun = mlrpinv) ;  # less robust

## Transfer of new spectra X1val 
## expected to be close to X2val
pred = predict(fm, X1val).pred

i = 1
f = Figure(size = (500, 300))
ax = Axis(f[1, 1])
lines!(X2val[i, :]; label = "x2")
lines!(ax, X1val[i, :]; label = "x1")
lines!(pred[i, :]; linestyle = :dash, label = "x1_corrected")
axislegend(position = :rb, framevisible = false)
f
```
""" 
function calds(X1, X2; fun = plskern, 
        kwargs...)
    fm = fun(X1, X2; kwargs...)
    CalDs(fm)
end

"""
    predict(object::CalDs, X; kwargs...)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `kwargs` : Optional arguments.
""" 
function predict(object::CalDs, X; 
        kwargs...)
    predict(object.fm, X; kwargs...)
end

