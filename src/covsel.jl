"""
    covsel(X, Y; nlv = nothing)
    covsel!(X::Matrix, Y::Matrix; nlv = nothing)
Variable (feature) selection from partial covariance (Covsel).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `nlv` : Nb. variables to select.

Covsel variable selection method (Roger et al. 2011). The selection is 
sequential. Once a variable is selected, `X` and `Y` are orthogonolized 
(deflated) to this variable, and a new variable (the one showing the maximum 
value for the criterion) is selected.

if `Y` is multivariate (q > 1), each column of `Y` is scaled by its 
uncorrected standard deviation. This gives the same scales to the columns
when computing the selection criterion.

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

res = covsel(X, y; nlv = 10) ;
res.sel
res.cov2

scatter(sqrt.(res.cov2), 
    axis = (xlabel = "Variable", ylabel = "Importance"))
```
""" 
function covsel(X, Y; nlv = nothing)
    covsel!(copy(ensure_mat(X)), copy(ensure_mat(Y)); 
        nlv = nlv)
end

function covsel!(X::Matrix, Y::Matrix; nlv = nothing) 
    n, p = size(X)
    q = nco(Y)
    isnothing(nlv) ? nlv = p : nothing 
    xmeans = colmean(X)
    ymeans = colmean(Y)   
    center!(X, xmeans)
    center!(Y, ymeans)
    if q > 1
        scale!(Y, colstd(Y))
    end
    xsstot = sum(X.^2)
    ysstot = sum(Y.^2)
    xss = zeros(nlv)
    yss = zeros(nlv)
    sel = Int64.(zeros(nlv))
    selcov = zeros(nlv)
    cov2 = zeros(p)
    H = similar(X, n, n)
    XtY = similar(X, p, q)
    for i = 1:nlv
        XtY .= X' * Y
        z = rowsum(XtY.^2) ./ n^2
        sel[i] = argmax(z)
        selcov[i] = z[sel[i]]
        cov2[sel[i]] = z[sel[i]]
        x = vcol(X, sel[i])
        H .= x * x' / dot(x, x)
        X .-= H * X 
        Y .-= H * Y
        xss[i] = sum(X.^2)
        yss[i] = sum(Y.^2)
    end
    cumpvarx = 1 .- xss ./ xsstot
    cumpvary = 1 .- yss ./ ysstot
    explvar = DataFrame((sel = sel, cov2 = selcov,
        cumpvarx = cumpvarx, cumpvary = cumpvary))
    (sel = sel, explvar, cov2)
end

