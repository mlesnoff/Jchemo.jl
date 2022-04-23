struct CalTransfDs
    fm
end


"""
    caltransf_ds(X1, X2; fun = mlrpinv, kwargs...)
Calibration transfer of spectral data with direct standardization (DS).
* `X1` : Target (standart) X-data (n, p).
* `X2` : X-data (n, p) to transfer to the standart.
* `fun` : Function used for fitting the transfer model.  
* `kwargs` : Optional arguments for `fun`.

`X1` and `X2` are assumed to represent the same n samples. 

## References

Y. Wang, D. J. Veltkamp, and B. R. Kowalski, “Multivariate Instrument Standardization,” 
Anal. Chem., vol. 63, no. 23, pp. 2750–2756, 1991, doi: 10.1021/ac00023a016.

## Examples
```julia
using JLD2, CairoMakie
mypath = joinpath(@__DIR__, "..", "data")
db = string(mypath, "\\", "caltransfer.jld2") 
@load db dat
pnames(dat)

X1cal = dat.X1cal
X2cal = dat.X2cal
X1val = dat.X1val
X2val = dat.X2val
n = nro(X1cal)
m = nro(X1val)

fm = caltransf_ds(X1cal, X2cal ; fun = mlrpinv) ;
#fm = caltransf_ds(X1cal, X2cal ; fun = plskern, nlv = 15) ;
pred = Jchemo.predict(fm, X2val).pred
i = 1
f, ax = lines(X1val[i, :])
lines!(X2val[i, :])
lines!(pred[i, :], linestyle = "--")
f
```
""" 
function caltransf_ds(X1, X2; fun = mlrpinv, kwargs...)
    fm = fun(X2, X1; kwargs...)
    CalTransfDs(fm)
end

"""
    predict(object::CalTransfDs, X; kwargs...)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `kwargs` : Optional arguments.
""" 
function predict(object::CalTransfDs, X; kwargs...)
    predict(object.fm, X; kwargs...)
end


