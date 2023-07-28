"""
    covsel(X, Y, weights = ones(nro(X)); 
        nlv = nothing, scal::Bool = false)
    covsel!(X::Matrix, Y::Matrix, weights = ones(nro(X)); 
        nlv = nothing, scal::Bool = false)
Variable (feature) selection from partial covariance (Covsel).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. variables to select.
* `scal` : Boolean. If `true`, each column of `X`
    is scaled by its uncorrected standard deviation.

Covsel method for variable selection (Roger et al. 2011). 
    
The selection is sequential. One first variable is selected (the variable
maximizing the selection criterion), `X` and `Y` are orthogonolized (deflated) 
to this variable, and a next variable is selected. And so on.

`Y` is automatically internally scaled by its uncorrected standard deviation.
This has the advantage to give the same importance to each `Y`-variable
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
function covsel(X, Y, weights = ones(nro(X)); 
        nlv = nothing, scal::Bool = false)
    covsel!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; 
        nlv = nlv, scal = scal)
end

function covsel!(X::Matrix, Y::Matrix, weights = ones(nro(X)); 
        nlv = nothing, scal::Bool = false) 
    n, p = size(X)
    q = nco(Y)
    isnothing(nlv) ? nlv = p : nothing 
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)  
    xscales = ones(p)
    if scal 
        xscales .= colstd(X)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end    
    yscales = colstd(Y, weights)
    #cscale!(Y, ymeans, yscales)
    xss = similar(X, nlv)
    yss = copy(xss)
    selcov = copy(xss)
    cov2 = zeros(p)
    sel = Int64.(zeros(nlv))
    H = similar(X, n, n)
    XtY = similar(X, p, q)
    D = Diagonal(weights)
    xsstot = sum(weights' * X.^2)
    ysstot = sum(weights' * Y.^2)
    for i = 1:nlv
        XtY = X' * (D * Y)  
        z = rowsum(XtY.^2)
        sel[i] = argmax(z)
        selcov[i] = z[sel[i]]
        cov2[sel[i]] = z[sel[i]]
        x = vcol(X, sel[i])
        H .= x * x' * D / dot(weights .* x, x)
        X .-= H * X 
        Y .-= H * Y
        xss[i] = sum(weights' * X.^2)
        yss[i] = sum(weights' * Y.^2)
    end
    cumpvarx = 1 .- xss ./ xsstot
    cumpvary = 1 .- yss ./ ysstot
    explvar = DataFrame((sel = sel, cov2 = selcov,
        cumpvarx = cumpvarx, cumpvary = cumpvary))
    (sel = sel, explvar, cov2, xmeans, xscales, 
        ymeans, yscales, weights)
end

