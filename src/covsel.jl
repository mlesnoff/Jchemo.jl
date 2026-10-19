"""
    covsel(; kwargs...)
    covsel(X, Y; kwargs...)
    covsel(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    covsel!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
Variable (feature) selection from partial covariance (Covsel).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Internally normalized to sum to 1.
Keyword arguments:
* `nlv` : Nb. variables to select.
* `scal` : Symbol defining the column scaling of `X` and `Y`. Possible values are: `:none` (default), `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

This is the Covsel algorithm described in Roger et al. 2011 for variable selection (see also 
Höskuldsson, A., 1992).

The selection is sequential and based on the *partial* covariance principle. One first variable in `X`
is selected: the one that maximizes the squared covariance with `Y` (if `Y` is multivariate, i.e. q > 1, 
this is the sum of the squared covariances over the columns of `Y`). Then `X` and `Y` are orthogonolized 
(deflated) to this variable. The selection criterion is recomputed and the next variable in (deflated) `X` 
is selected, and so on.

When `Y` is multivariate, it is recommended to use scaling to give the same importance to the variables 
when computing the covariances.

**Note:** A faster alternative to function `covsel` is function `splsr` (with `nlv = 1`) that uses 
the kernel PLS algorithm of Dayal & McGregor 1997. Another faster algorithm is described in Mishra 2022.   

# References
Höskuldsson, A., 1992. The H-principle in modelling with applications to chemometrics. Chemometrics 
and Intelligent Laboratory Systems, Proceedings of the 2nd Scandinavian Symposium on Chemometrics 14, 
139–153. https://doi.org/10.1016/0169-7439(92)80099-P

Mishra, P., 2022. A brief note on a new faster covariate’s selection (fCovSel) algorithm. 
Journal of Chemometrics 36, e3397. https://doi.org/10.1002/cem.3397

Roger, J.M., Palagos, B., Bertrand, D., Fernandez-Ahumada, E., 2011. covsel: Variable selection for highly 
multivariate and multi-response calibration: Application to IR spectroscopy. 
Chem. Lab. Int. Syst. 106, 216-223.

Wikipedia
https://en.wikipedia.org/wiki/Partial_correlation

# Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat

X = dat.X
y = dat.Y.tbc
wlst = names(X)
wl = parse.(Float64, wlst)

nlv = 10
model = covsel(; nlv)
fit!(model, X, y)
fitm = model.fitm ;
@names fitm

fitm.sel    # indexes of selected variables for each LV
wl[fitm.sel]

fitm.selc   # associated squared covariances

@head fitm.C  # square covariance profile for each LV

i = 1         # profile for LV 'i'
xticks = 400:100:2500
f, ax = plotsp(fitm.C[i:i, :], wl; size = (900, 300), xticks, title = string("LV ", i), 
    xlabel = "Wawelength (nm)", ylabel = "Cov^2")   
vlines!(ax, wl[fitm.sel[i]])
f

@head transf(model, X)
@head transf(model, X, 3)

res = summary(model)
res.explvarx   # X-variance explained
res.explvary   # Y-variance explained

plotxy(1:nlv, fitm.selc; xlabel = "Variable", ylabel = "Importance").f
```
""" 
covsel(; kwargs...) = JchemoModel(covsel, nothing, kwargs)

function covsel(X, Y; kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = pweight(ones(eltype(X), nro(X)))
    covsel(X, Y, weights; kwargs...)
end

function covsel(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    covsel!(copy(X), copy(Y), weights; kwargs...)
end

function covsel!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    par = recovkw(ParCovsel, kwargs).par
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, par.nlv)
    par.nlv = nlv  
    # Centering/scaling of Y
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)  
    fcenter!(X, xmeans)
    fcenter!(Y, ymeans)
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal != :none
        colscal = def_colscal(par.scal) 
        xscales .= colscal(X, weights)
        yscales .= colscal(Y, weights)
        fscale!(X, xscales)
        fscale!(Y, yscales)
    end
    # End
    sqrtw = sqrt.(weights.values)
    fweightr!(X, sqrtw)
    fweightr!(Y, sqrtw)
    xsstot = frob2(X)
    ysstot = frob2(Y)
    # Pre-allocation
    sel = zeros(Int, nlv)
    selc = similar(X, nlv)
    x = similar(X, n, 1)
    XtY2 = similar(X, p, q)
    Kx = similar(X, 1, p)
    Ky = similar(X, 1, q)
    c = similar(X, p)
    C = similar(X, nlv, p)
    xss = similar(selc)
    yss = similar(selc)
    # end
    for i = 1:nlv
        XtY2 .= X' * Y   # (p, q)
        @. XtY2 = XtY2 * XtY2 
        if q == 1
            c .= XtY2
        else
            c .= rowsum(XtY2)
        end
        sel[i] = argmax(c)
        selc[i] = c[sel[i]]
        C[i, :] .= c
        # Orthogonal projection of X and Y on variable x 
        # and deflations X - X_hat and Y - Y_hat 
        # P_proj (n, n) = x * inv(x' * x) * x' = x * x' / dot(x, x)
        # x * x' is heavy ==> not computed
        x .= vcol(X, sel[i])
        Kx .= x' * X
        Ky .= x' * Y
        dotx = dot(x, x) 
        X .-= x * Kx / dotx   
        Y .-= x * Ky / dotx
        # End
        xss[i] = frob2(X)
        yss[i] = frob2(Y)
    end
    Covsel(sel, selc, C, xss, yss, xsstot, ysstot, xmeans, xscales, ymeans, yscales, weights, par)
end

""" 
    transf(object::Covsel, X, nlv::Int)
Compute selected variables from a fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which the selected variables are computed.
* `nlv` : Nb. variables to compute.
""" 
transf(object::Covsel, X) = X[:, object.sel]

function transf(object::Covsel, X, nlv::Int)
    a = object.par.nlv
    nlv = isnothing(nlv) ? a : min(nlv, a)
    X[:, object.sel[1:nlv]]
end

"""
    summary(object::Covsel)
Summarize the fitted model.
* `object` : The fitted model.
""" 
function Base.summary(object::Covsel)
    nlv = length(object.sel)
    xsstot = object.xsstot
    ysstot = object.ysstot
    cumpvarx = 1 .- object.xss / xsstot
    cumpvary = 1 .- object.yss / ysstot
    pvarx = cumpvarx - [0; cumpvarx[1:(nlv - 1)]]
    pvary = cumpvary - [0; cumpvary[1:(nlv - 1)]]
    vnlv = collect(1:nlv)
    explvarx = DataFrame(nlv = vnlv, sel = object.sel, pvar = pvarx, cumpvar = cumpvarx)
    explvary = DataFrame(nlv = vnlv, sel = object.sel, pvar = pvary, cumpvar = cumpvary)
    (explvarx = explvarx, explvary)
end


