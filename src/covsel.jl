"""
    covsel(X, Y; nvar = nothing, scaly = true)
Variable (feature) selection with the covsel method
* `X` : X-data.
* `Y` : Y-data.
* `nvar` : Nb. variables to select.
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
""" 
function covsel(X, Y; nvar = nothing, scaly = true)
    covsel!(copy(X), copy(Y); nvar = nvar, scaly = scaly)
end

function covsel!(X, Y; nvar = nothing, scaly = true) 
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    isnothing(nvar) ? nvar = p : nothing 
    xmeans = colmeans(X) 
    ymeans = colmeans(Y)   
    center!(X, xmeans)
    center!(Y, ymeans)
    if scaly
        yscales = sqrt.(sum(Y.^2; dims = 1))
        Y = scale(Y, yscales)
    end
    xsstot = sum(X.^2)
    ysstot = sum(Y.^2)
    xss = zeros(nvar)
    yss = zeros(nvar)
    selvar = Int64.(zeros(nvar))
    covs = zeros(p)
    P = similar(X, n, n)
    for i = 1:nvar
        z = vec(sum(abs.(X' * Y); dims = 2))
        zsel = argmax(z)
        selvar[i] = zsel
        covs[zsel] = z[zsel]
        u = vcol(X, zsel)
        P .= u * u' / sum(u.^2)
        X .= X .- P * X 
        Y .= Y .- P * Y
        xss[i] = sum(X.^2)
        yss[i] = sum(Y.^2)
    end
    cumpvarx = 1 .- xss / xsstot
    cumpvary = 1 .- yss / ysstot
    sel = DataFrame((sel = selvar, 
        cumpvarx = cumpvarx, cumpvary = cumpvary))
    (sel = sel, covs = covs)
end




