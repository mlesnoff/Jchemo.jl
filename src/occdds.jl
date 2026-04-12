"""
    occdds(; kwargs...)
    occdds(object, X; kwargs...)
One-class classification (OCC) using DD-Simca.
* `fitm` : The preliminary model (e.g., object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the reference (= target) class.
* `X` : Training X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `fcentr` : A function that computes the centers of the empirical distributions of the squared score and orthogonal 
    distances (SD2 and OD2). By default, `fcentr = meanv`.
* `fscal` : A function that computes the scales of the empirical distributions of the squared score and orthogonal 
    distances (SD2 and OD2). By default, `fcentr = stdv`.
* `alpha` : Risk-I level to compute the quantile (re-scaled Chi-2) of the consensus variable.

In this function, outlierness `d` of a given observation is a consensus between the squared score distance (SD2) and the
squared orthogonal distance (OD2), defined by: 
* d = (nu1 / mu1) * SD2 + (nu2 / mu2) * SD2.
The empirical training SD2 and OD2 distributions are assumed to approximately follow independent Chi-2s. Parameters 
{mu1, mu2} represent the respective distribution centers, and {nu1, nu2} the respective dofs, estimated by the moments 
method. Outlierness `d` for the training set is assumed to approximatively follow a Chi-2(nu) distribution 
where nu = nu1 + nu2.

## References
Kucheryavskiy, S., Rodionova, O., Pomerantsev, A., 2024. A comprehensive tutorial on Data-Driven SIMCA: Theory 
and implementation in web. Journal of Chemometrics 38, e3556. https://doi.org/10.1002/cem.3556

Pomerantsev, A.L., 2008. Acceptance areas for multivariate classification derived by projection methods. 
Journal of Chemometrics 22, 601–609. https://doi.org/10.1002/cem.1147

Pomerantsev, A.L., Rodionova, O.Y., 2014. Concept and role of extreme objects in PCA/SIMCA. 
Journal of Chemometrics 28, 429–438. https://doi.org/10.1002/cem.2506

Rodionova, O., Kucheryavskiy, S., Pomerantsev, A., 2021. Efficient tools for principal component analysis of complex 
data — a tutorial. Chemometrics and Intelligent Laboratory Systems 213, 104304. 
https://doi.org/10.1016/j.chemolab.2021.104304

## Examples
```julia
```
""" 
#occdds(; kwargs...) = JchemoModel(occdds, nothing, kwargs)

function occdds(fitm, X; fcentr = meanv, fscal = stdv, alpha = .05) 
    #par = recovkw(ParOcc, kwargs).par 
    sd = outsd(fitm).d
    od = outod(fitm, X).d
    ##
    d = sd.^2 
    mu = fcentr(d)
    sigma = fscal(d)
    g = sigma^2 / (2 * mu)
    nu = 2 * (mu / sigma)^2
    nu = max(1, round(Int, nu))
    cutoff = mu / nu * quantile(Chisq(nu), 1 - alpha)
    sd2 = (d = d, mu, sigma, g, nu, cutoff)
    #quantile(Chisq(nlv), 1 - alpha)
    ##
    d = od.^2 
    mu = fcentr(d)
    sigma = fscal(d)
    g = sigma^2 / (2 * mu)
    nu = 2 * (mu / sigma)^2
    nu = max(1, round(Int, nu))
    cutoff = mu / nu * quantile(Chisq(nu), 1 - alpha)
    od2 = (d = d, mu, sigma, g, nu, cutoff)
    ##
    d = sd2.nu / sd2.mu * sd2.d + od2.nu / od2.mu * od2.d 
    nu = sd2.nu + od2.nu
    cutoff = quantile(Chisq(nu), 1 - alpha)
    a = 1 / od2.nu * cutoff 
    b = sd2.nu / od2.nu 
    ## 
    (d = d, nu, cutoff, sd2, od2, a, b)
end

"""
    predict(object::Occdds, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict_occds(object, X)
    m = nro(X)
    sd = predict(object.fitmsd, X).d
    od = predict(object.fitmod, X).d
    nam = string.(names(sd), "_sd")
    rename!(sd, nam)
    nam = string.(names(od), "_od")
    rename!(od, nam)
    d = hcat(sd, od)
    d.dstand = [sqrt(sd.dstand_sd[i] * od.dstand_od[i]) for i in eachindex(sd.dstand_sd)]
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i in eachindex(d.d)]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

