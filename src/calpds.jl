"""
    calpds(X, Xtar; npoint = 5, 
        fun = mlrpinv, kwargs...)
Piecewise direct standardization (PDS) for calibration transfer of spectral data.
* `X` : Spectra to transfer to the target (n, p).
* `Xtar` : Target spectra (n, p).
* `npoint` : Half-window size (nb. points left/right to the target wavelength) 
* `fun` : Function used as transfer model.  
* `kwargs` : Optional arguments for `fun`.

`Xtar` and `X` must represent the same n standard samples.

The objective is to transform spectra `X` to new spectra as close 
as possible as the target `Xtar`. Method PDS fits models 
(defined in `fun`) that predict `Xtar` from `X`.

The window used in `X` to predict wavelength "i" in `Xtar` is:

* i - `npoint`, i - `npoint` + 1, ..., i, ..., i + `npoint` - 1, i + `npoint`

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
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/caltransfer.jld2")
@load db dat
pnames(dat)

## Target
Xtarcal = dat.X1cal
Xtarval = dat.X1val
## To transfer to the target
Xcal = dat.X2cal
Xval = dat.X2val
n = nro(Xtarcal)
m = nro(Xtarval)

fm = calpds(Xcal, Xtarcal;  npoint = 2, 
    fun = plskern, nlv = 1) ;
## Transfered (= corrected) spectra, 
## expected to be close to Xtarval
pred = Jchemo.predict(fm, Xval).pred

i = 1
f = Figure(size = (500, 300))
ax = Axis(f[1, 1])
lines!(Xtarval[i, :]; label = "xtar")
lines!(ax, Xval[i, :]; label = "x_not_correct")
lines!(pred[i, :]; linestyle = :dash, label = "x_correct")
axislegend(position = :rb, framevisible = false)
f
```
""" 
function calpds(X, Xtar; npoint = 5, 
        fun = mlrpinv, kwargs...)
    p = nco(X)
    fm = list(p)
    s = list(p)
    npo = repeat([npoint], p)
    npo[1:npoint] .= collect(1:npoint) .- 1
    npo[(p - npoint + 1):p] .= collect(npoint:-1:1) .- 1
    @inbounds for i = 1:p
        s[i] = collect((i - npo[i]):(i + npo[i]))
        fm[i] = fun(vcol(X, s[i]), vcol(Xtar, i); kwargs...)
    end
    CalPds(fm, s)
end

"""
    predict(object::CalPds, X; kwargs...)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `kwargs` : Optional arguments.
""" 
function predict(object::CalPds, X; kwargs...)
    X = ensure_mat(X)
    m, p = size(X)
    pred = similar(X, m, p)
    @inbounds for i = 1:p 
        pred[:, i] .= predict(object.fm[i], 
            vcol(X, object.s[i]); kwargs...).pred
    end
    (pred = pred,)
end

