"""
    covsel(X, Y; nlv = nothing, typ = "cov")
    covsel!(X::Matrix, Y::Matrix; nlv = nothing, typ = "cov")
Variable (feature) selection from partial correlation or covariance (Covsel).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `nlv` : Nb. variables to select.
* `typ` : Criterion used at each variable selection. 
    Possible values are: "cov" (squared covariance with `Y`, such as in Roger 
    et al. 2011) and "cor" (squared correlation with `Y`).

The selection is sequential. Once a variable is selected, 
`X` and `Y` are orthogonolized to this variable, 
and a new variable (the one showing the maximum value for the criterion)
is selected.

if `Y` is multivariate (q > 1), each column of `Y` is scaled by its 
uncorrected stnandard deviation. This gives the same scales to the columns
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
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
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
function covsel(X, Y; nlv = nothing, typ = "cov")
    covsel!(copy(ensure_mat(X)), copy(ensure_mat(Y)); 
        nlv = nlv, typ = typ)
end

function covsel!(X::Matrix, Y::Matrix; nlv = nothing, 
        typ = "cov") 
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
    selvar = Int64.(zeros(nlv))
    selcov = zeros(nlv)
    cov2 = zeros(p)
    H = similar(X, n, n)
    zcov = similar(X, p, q)
    C = similar(X, p, nlv)
    for i = 1:nlv
        if typ == "cov"
            zcov .= cov(X, Y; corrected = false)
            z = rowsum(zcov.^2)
        end
        if typ == "cor"
            zcov .= cor(X, Y)
            z = rowsum(zcov.^2)
        end
        ## Same result as "cor"
        ## ==> not useful
        if typ == "aic"           
            zscor = zeros(p)
            for j = 1:p
                x = vcol(X, j)
                fm = mlr(x, Y; noint = true) 
                pred = Jchemo.predict(fm, x).pred
                resid = residreg(pred, Y)
                df = 2
                zscor[j] = log.(sum(resid.^2)) + q * 2 * df / n
            end
            z = -zscor
        end
        ## End
        C[:, i] .= z
        zsel = argmax(z)
        selvar[i] = zsel
        selcov[i] = z[zsel]
        cov2[zsel] = z[zsel]
        x = vcol(X, zsel)
        H .= x * x' / dot(x, x)
        X .-= H * X 
        Y .-= H * Y
        xss[i] = sum(X.^2)
        yss[i] = sum(Y.^2)
    end
    cumpvarx = 1 .- xss / xsstot
    cumpvary = 1 .- yss / ysstot
    sel = DataFrame((sel = selvar, cov2 = selcov,
        cumpvarx = cumpvarx, cumpvary = cumpvary))
    (sel = sel, cov2, C)
end

