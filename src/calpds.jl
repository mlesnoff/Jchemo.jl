struct CalPds
    fm
    s
end

"""
    calpds(Xt, X; fun = mlrpinv, m = 5, kwargs...)
Piecewise direct targetization (PDS) for calibration transfer of spectral data.
* `Xt` : Target spectra, (n, p).
* `X` : Spectra to transfer to the target, (n, p).
* `fun` : Function used for fitting the transfer model.  
* `m` : Half-window size (nb. points left/right to the target wavelength) 
* `kwargs` : Optional arguments for `fun`.

`Xt` and `X` must represent the same n samples.

The objective is to transform spectra `X` to spectra as close 
as possible as the target `Xt`. The principle of the method is to fit models 
predicting `Xt` from `X.

To predict wavelength i in `Xt`, the window used in `X` is :

* i - m, i - m + 1, ..., i, ..., i + m - 1, i + m

## References
Bouveresse, E., Massart, D.L., 1996. Improvement of the piecewise direct targetisation procedure 
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

## Target
Xtcal = dat.X1cal
Xtval = dat.X1val
## To be transfered
Xcal = dat.X2cal
Xval = dat.X2val

n = nro(Xtcal)
m = nro(Xtval)

fm = calpds(Xtcal, Xcal; fun = plskern, nlv = 1, m = 2) ;
## Transferred data
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
function calpds(Xt, X; fun = mlrpinv, m = 5, kwargs...)
    p = nco(Xt)
    fm = list(p)
    s = list(p)
    zm = repeat([m], p)
    zm[1:m] .= collect(1:m) .- 1
    zm[(p - m + 1):p] .= collect(m:-1:1) .- 1
    @inbounds for i = 1:p
        s[i] = collect((i - zm[i]):(i + zm[i]))
        fm[i] = fun(X[:, s[i]], Xt[:, i]; kwargs...)
    end
    CalPds(fm, s)
end

"""
    predict(object::CalPds, X; kwargs...)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : Spectra to transfer to target form.
* `kwargs` : Optional arguments.
""" 
function predict(object::CalPds, X; kwargs...)
    X = ensure_mat(X)
    m, p = size(X)
    pred = similar(X, m, p)
    for i = 1:p 
        pred[:, i] .= predict(object.fm[i], X[:, object.s[i]]; kwargs...).pred
    end
    (pred = pred,)
end



