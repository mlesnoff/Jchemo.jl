"""
    covsel(; kwargs...)
    covsel(X, Y; kwargs...)
    covsel(X, Y, weights::Weight; kwargs...)
    covsel!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Variable (feature) selection from partial covariance/correlation (Covsel).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Internally normalized to sum to 1.
* `nlv` : Nb. variables to select.
* `scalx` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.
* `scaly` : Boolean. If `true`, each column of `Y` is scaled by its uncorrected standard deviation.

This is the Covsel algorithm described in Roger et al. 2011 for variable selection. Function `covsel` also 
proposes an option not present in the original algorithm: correlation can be used instead of covariance to 
compute the selection criterion.

The selection is sequential and based on the *partial* covariance/correlation principle. One first variable is 
selected (the variable maximizing the selection criterion: squared partail covariance/correlation), `X` and `Y` 
are orthogonolized (deflated) to this variable, the selection criterion is recomputed and the next variable 
is selected. And so on.

The prelimianry scaling of `X` is optional. In contrast, `Y` is automatically internally scaled by its 
uncorrected standard deviation. For the covariance This has the advantage to give the same importance to each `Y`-variable
when `Y`is  multivariate (q > 1). This has no effect when `Y` is 
univariate.

## References
Höskuldsson, A., 1992. The H-principle in modelling with applications 
to chemometrics. Chemometrics and Intelligent Laboratory Systems, 
Proceedings of the 2nd Scandinavian Symposium on Chemometrics 14, 
139–153. https://doi.org/10.1016/0169-7439(92)80099-P

Roger, J.M., Palagos, B., Bertrand, D., Fernandez-Ahumada, E., 2011. 
covsel: Variable selection for highly multivariate and multi-response 
calibration: Application to IR spectroscopy. 
Chem. Lab. Int. Syst. 106, 216-223.

Wikipedia
https://en.wikipedia.org/wiki/Partial_correlation

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
y = dat.Y.tbc

fm = covsel(X, y; nlv = 10) ;
fm.sel
fm.cov2

scatter(sqrt.(fm.cov2), 
    axis = (xlabel = "Variable", ylabel = "Importance"))
```
""" 
covsel(; kwargs...) = JchemoModel(covsel, nothing, kwargs)

function covsel(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    covsel(X, Y, weights; kwargs...)
end

function covsel(X, Y, weights::Jchemo.Weight; kwargs...)
    covsel!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function covsel!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
    par = recovkw(ParCovsel, kwargs).par
    Q = eltype(X)
    ## Specific for Da functions
    isa(Y, BitMatrix) ? Y = convert.(Q, Y) : nothing
    ## End 
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, par.nlv)  
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)  
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scalx
        xscales .= colstd(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    if par.scaly
        yscales .= colstd(Y, weights)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(Y, ymeans)
    end
    sqrtw = sqrt.(weights.w)
    fweight!(X, sqrtw)
    fweight!(Y, sqrtw)
    xsstot = frob2(X)
    ysstot = frob2(Y)
    ##
    sel = zeros(Int, nlv)
    selc = similar(X, nlv)
    x = similar(X, n, 1)
    XtY = similar(X, p, q)
    c = similar(X, p)
    xss = copy(selc)
    yss = copy(selc)
    for i = 1:nlv
        if par.meth == :cor
            xscales .= colnorm(X)
            yscales .= colnorm(Y)
            if i > 1
                xscales[sel[1:(i - 1)]] .= 1  # remove divisions by zeros
            end
            fscale!(X, xscales)
            fscale!(Y, yscales)
        end             
        XtY .= X' * Y   # (p, q) 
        if q == 1
            c .= XtY.^2
        else
            c .= rowsum(XtY.^2)
        end
        sel[i] = argmax(c)
        selc[i] = c[sel[i]]
        x .= vcol(X, sel[i])
        dotx = dot(x, x) 
        ## Projecion matrix on x (n, n) = x * inv(x' * x) * x' = x * x' / dot(x, x)
        X .-= x * x' * X / dotx   
        Y .-= x * x' * Y / dotx 
        xss[i] = frob2(X)
        yss[i] = frob2(Y)
    end
    Covsel(sel, selc, xss, yss, xsstot, ysstot, xmeans, xscales, ymeans, yscales, weights, par)
end

""" 
    transf(object::Covsel, X; nlv = nothing)
Compute selected variables from a fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which the selected variables are computed.
* `nlv` : Nb. variables to compute.
""" 
function transf(object::Covsel, X; nlv = nothing)
    a = length(object.sel)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
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
    explvarx = DataFrame(nlv = 1:nlv, sel = object.sel, pvar = pvarx, cumpvar = cumpvarx)
    explvary = DataFrame(nlv = 1:nlv, sel = object.sel, pvar = pvary, cumpvar = cumpvary)
    (explvarx = explvarx, explvary)
end


