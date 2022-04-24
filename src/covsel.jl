"""
    covsel(X, Y; nlv = nothing, scaly = true)
    covsel!(X::Matrix, Y::Matrix; nlv = nothing, scaly = true)
Variable (feature) selection with the covsel method
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `nlv` : Nb. variables to select.
* `scaly` : If `true`, columns of `Y` are scaled.

The function alse returns the absolute partial covariances (`covs`)
for the selected variables.   

## References
Höskuldsson, A., 1992. The H-principle in modelling with applications 
to chemometrics. Chemometrics and Intelligent Laboratory Systems, 
Proceedings of the 2nd Scandinavian Symposium on Chemometrics 14, 
139–153. https://doi.org/10.1016/0169-7439(92)80099-P

Roger, J.M., Palagos, B., Bertrand, D., Fernandez-Ahumada, E., 2011. 
covsel: Variable selection for highly multivariate and multi-response 
calibration: Application to IR spectroscopy. 
Chem. Lab. Int. Syst. 106, 216-223.

## Examples
```julia
using JLD2, CairoMakie
mypath = joinpath(@__DIR__, "..", "data")
db = string(mypath, "\\", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X
y = dat.Y.y

res = covsel(X, y; nlv = 20) ;
res.sel
res.covs

scatter(res.covs, 
    axis = (xlabel = "Variable", ylabel = "Importance"))
```
""" 
function covsel(X, Y; nlv = nothing, scaly = true)
    covsel!(copy(ensure_mat(X)), copy(ensure_mat(Y)); 
        nlv = nlv, scaly = scaly)
end

function covsel!(X::Matrix, Y::Matrix; nlv = nothing, scaly = true) 
    n, p = size(X)
    isnothing(nlv) ? nlv = p : nothing 
    xmeans = colmean(X) 
    ymeans = colmean(Y)   
    center!(X, xmeans)
    center!(Y, ymeans)
    if scaly
        yscales = sqrt.(sum(Y.^2; dims = 1))
        Y = scale(Y, yscales)
    end
    xsstot = sum(X.^2)
    ysstot = sum(Y.^2)
    xss = zeros(nlv)
    yss = zeros(nlv)
    selvar = Int64.(zeros(nlv))
    selcov = zeros(nlv)
    covs = zeros(p)
    H = similar(X, n, n)
    for i = 1:nlv
        z = vec(sum(abs.(X' * Y); dims = 2))
        zsel = argmax(z)
        selvar[i] = zsel
        selcov[i] = z[zsel]
        covs[zsel] = z[zsel]
        x = vcol(X, zsel)
        H .= x * x' / sum(x.^2)
        X .= X .- H * X 
        Y .= Y .- H * Y
        xss[i] = sum(X.^2)
        yss[i] = sum(Y.^2)
    end
    cumpvarx = 1 .- xss / xsstot
    cumpvary = 1 .- yss / ysstot
    sel = DataFrame((sel = selvar, cov = selcov,
        cumpvarx = cumpvarx, cumpvary = cumpvary))
    (sel = sel, covs = covs)
end




