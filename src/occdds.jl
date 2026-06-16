"""
    occdds(; kwargs...)
    occdds(object, X; kwargs...)
One-class classification (OCC) using DD-Simca.
* `fitm` : The preliminary model (e.g., object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the reference (= target) class.
* `X` : Training X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `fcentr` : A function that computes the centers of the empirical distributions of the squared score and orthogonal 
    distances (SD^2 and OD^2). By default, `fcentr = meanv`.
* `fscal` : A function that computes the scales of the empirical distributions of SD^2 and OD^2. By default, `fcentr = stdv`.
* `alpha` : Risk-I level to compute the parametric quantile (re-scaled Chi-2) of the consensus variable.

The function implements OCC based on the outlierness `d` as defined in the DD-Simca method. The principle is described 
below.

In DD-Simca, SD^2 and OD^2 on the training set are assumed to have approximately independent re-scaled Chi-2 distributions, 
as follows:
* SD^2 ~ g1 * Chi-2(nu1)
* OD^2 ~ g2 * Chi-2(nu2)
where {nu1, nu2} and {g1, g2} are Chi-2 degrees of freedom (dofs) and scaling parameters, respectively. 
This is equivalent to assume Chi-2 distributions for the scaled SD^2 and OD^2, as follows:
* (1 / g1) * SD^2 ~ Chi-2(nu1)
* (1 / g2) * OD^2 ~ Chi-2(nu2)
or, by using the properties of the Chi-2 distribution (see in **Details** below),
* (nu1 / mu1) * SD^2 ~ Chi-2(nu1)
* (nu2 / mu2) * OD^2 ~ Chi-2(nu2)
where parameters {mu1, mu2} represent centers of the SD^2 and OD^2 training distributions.

Outlierness `d` of a given observation is finally defined by the following consensus: 
* `d` = (1 / g1) * SD^2 + (1 / g2) * OD^2
or, equivalently, by
* `d` = (nu1 / mu1) * SD^2 + (nu2 / mu2) * OD^2
where `d` is assumed, for the training set, to approximately follow a Chi-2 distribution with nu = nu1 + nu2 
dofs. Parameters {mu1, mu2} and {nu1, nu2} are estimated by the moments method on the training set. 
This assumed distribution is use to compute a parametric cutoff for `d` for a given risk-I level `alpha`.

**Details:**

Let us note Z to represent SD^2 or OD^2. The method assumes that Z ~ g * Chi-2(nu) or, equivalently, 
(1 / g) * Z ~ Chi-2(nu). If mu and sigma^2 represent the expectation and variance of Z (i.e., mu = E[Z] and sigma^2 = Var[Z]), 
it follows from the properties of the Chi-2 distribution that
* g = mu / nu = sigma^2 / (2 * mu)
* nu = 2 * (mu / sigma)^2 
Nomikos & MacGregor (1995) proposed, for OD^2, to estimate parameters {mu, sigma^2} (and therefore {g, nu}) by 
the moments method on the training set. This consists to estimate {mu, sigma^2} by the sample mean (or other center statistic) 
and variance (or other scale statistic), respectively, of the observed (training) distribution of Z.
In DD-Simca, the same method is applied to both SD^2 and OD^2. This allows to assume and compute the Chi-2 distribution
of the consensus variable `d`.

## References
Kucheryavskiy, S., Rodionova, O., Pomerantsev, A., 2024. A comprehensive tutorial on Data-Driven SIMCA: Theory 
and implementation in web. Journal of Chemometrics 38, e3556. https://doi.org/10.1002/cem.3556

Nomikos, P., MacGregor, J.F., 1995. Multivariate SPC Charts for Monitoring Batch Processes. null 37, 41–59. 
https://doi.org/10.1080/00401706.1995.10485888


Pomerantsev, A.L., 2008. Acceptance areas for multivariate classification derived by projection methods. 
Journal of Chemometrics 22, 601–609. https://doi.org/10.1002/cem.1147

Pomerantsev, A.L., Rodionova, O.Y., 2014. Concept and role of extreme objects in PCA/SIMCA. 
Journal of Chemometrics 28, 429–438. https://doi.org/10.1002/cem.2506

Rodionova, O., Kucheryavskiy, S., Pomerantsev, A., 2021. Efficient tools for principal component analysis of complex 
data — a tutorial. Chemometrics and Intelligent Laboratory Systems 213, 104304. 
https://doi.org/10.1016/j.chemolab.2021.104304

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/challenge2018.jld2") 
@load db dat
@names dat
X = dat.X    
Y = dat.Y
model = savgol(npoint = 21, deriv = 2, degree = 3)
fit!(model, X) 
Xp = transf(model, X) 
s = Bool.(Y.test)
Xtrain = rmrow(Xp, s)
Ytrain = rmrow(Y, s)
yclatrain = Ytrain.typ
Xtest = Xp[s, :]
Ytest = Y[s, :]
yclatest = Ytest.typ 

#### Build the data used in the example
## Training reference class (= target = 'in') is "EHH" 
s = yclatrain .== "EHH"
Xtrain_in = Xtrain[s, :]    
ntrain_in = nro(Xtrain_in)
## Observations 'in' to be predicted (should be predicted 'in')
s = yclatest .== "EHH"
Xtest_in = Xtest[s, :] 
ntest_in = nro(Xtest_in)
## Observations 'out' ("PEE") to be predicted (should be predicted 'out')
s = yclatest .== "PEE"
Xtest_out = Xtest[s, :] 
ntest_out = nro(Xtest_out)
## Only used to compute error rates
ntot = ntrain_in + ntest_in + ntest_out
(ntot = ntot, ntrain_in, ntest_in, ntest_out)
ytrain_in = fill("in", ntrain_in)
ytest_in = fill("in", ntest_in)
ytest_out = fill("out", ntest_out)

#### Fit a preliminary Pca model on the training data 'in'
nlv = 15
model0 = pcasvd(; nlv) 
#model0 = pcaout(; nlv) 
fit!(model0, Xtrain_in) 
fitm0 = model0.fitm ;
res = summary(model0, Xtrain_in).explvarx 
plotgrid(res.nlv, res.pvar; step = 2, xlabel = "Nb. LVs", ylabel = "% Variance explained").f
Ttrain_in = fitm0.T

#### To describe the data, project the test observations in the fitted score space 'in'
Ttest_in = transf(model0, Xtest_in)
Ttest_out = transf(model0, Xtest_out)
#GLMakie.activate!()   # requires GLMakie
T = vcat(Ttrain_in, Ttest_in, Ttest_out)
group = vcat(fill("Train_in", ntrain_in), fill("Test_in", ntest_in), fill("Test_out", ntest_out))
color = [:purple, (:green, .7), (:red, .3)]
i = 1
plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], group; color, leg_title = "Type of obs.", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1), zlabel = string("PC", i + 2)).f

#### Fit the Occ model based on the fitted score space 'in' 
model = occdds()
fit!(model, fitm0, Xtrain_in)
@names model 
fitm = model.fitm ;
@names fitm 
@head dtrain_in = fitm.d
cutoff = fitm.cutoff

d = dtrain_in.dstand
f, ax = plotxy(1:length(d), d; color = (:red, .3), size = (500, 300), xlabel = "Observation index", 
    ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
s = d .> 1
scatter!(ax, (1:length(d))[s], d[s]; color = :red)
f

d = dtrain_in.d
sd2mu = dtrain_in.sd2mu
od2mu = dtrain_in.od2mu
a = fitm.coefs[1]
b = fitm.coefs[2]
s = d .> cutoff
f, ax = plotxy(sd2mu, od2mu; xlabel = "SD2 / mu", ylabel = "OD2 / mu")
scatter!(ax, sd2mu[s], od2mu[s]; color = :red, label = "Extreme")
ablines!(ax, a, b; color = :red, linewidth = .7, linestyle = :dash)
axislegend(ax; position = :rb)
f

#### Predict the test observations 'in'
res = predict(model, Xtest_in) ;
@names res
@head pred = res.pred
@head dtest_in = res.d
tab(pred)
errp(pred, ytest_in)
conf(pred, ytest_in).cnt

#### Predict the test observations 'out'
res = predict(model, Xtest_out) ;
@names res
@head pred = res.pred
@head dtest_out = res.d
tab(pred)
errp(pred, ytest_out)
conf(pred, ytest_out).cnt

d = vcat(dtrain_in.dstand, dtest_in.dstand, dtest_out.dstand)
color = [:purple, (:green, .7), (:red, .3)]
f, ax = plotxy(1:length(d), d, group; color, size = (500, 300), leg_title = "Type of obs.", 
    title = "OD", xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f

d = dtrain_in.d
sd2mu = dtrain_in.sd2mu
od2mu = dtrain_in.od2mu
a = fitm.coefs[1]
b = fitm.coefs[2]
s = d .> cutoff
f, ax = plotxy(sd2mu, od2mu; xlabel = "SD2 / mu", ylabel = "OD2 / mu")
scatter!(ax, sd2mu[s], od2mu[s]; color = :red, label = "Test-In")
scatter!(ax, dtest_in.sd2mu, dtest_in.od2mu; color = (:purple, .5), label = "Test-In")
scatter!(ax, dtest_out.sd2mu, dtest_out.od2mu; color = :green, label = "Test-Out")
ablines!(ax, a, b; color = :red, linewidth = .7, linestyle = :dash)
axislegend(ax; position = :rb)
f
```
""" 
occdds(; kwargs...) = JchemoModel(occdds, nothing, kwargs)

function occdds(fitm, X; kwargs...) 
    par = recovkw(ParOccdds{Q}, kwargs).par 
    alpha = par.alpha
    @assert 0 <= alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."    
    nlv = nco(fitm.T) 
    sd = outsd(fitm)
    od = outod(fitm, X)
    alpha = par.alpha
    ## SD2
    d = sd.d.^2 
    mu = par.fcentr(d)
    sigma = par.fscal(d)
    g = sigma^2 / (2 * mu)
    nu = 2 * (mu / sigma)^2
    nu = max(1, round(Int, nu))
    cutoff = mu / nu * quantile(Chisq(nu), 1 - alpha)
    sd2 = (d = d, mu, sigma, g, nu, cutoff, tscales = sd.tscales)
    ## OD2
    d = od.d.^2 
    mu = par.fcentr(d)
    sigma = par.fscal(d)
    g = sigma^2 / (2 * mu)
    nu = 2 * (mu / sigma)^2
    nu = max(1, round(Int, nu))
    cutoff = mu / nu * quantile(Chisq(nu), 1 - alpha)
    od2 = (d = d, mu, sigma, g, nu, cutoff)
    ##
    nu = sd2.nu + od2.nu
    cutoff = quantile(Chisq(nu), 1 - alpha)
    d = sd2.nu / sd2.mu * sd2.d + od2.nu / od2.mu * od2.d 
    e_cdf = StatsBase.ecdf(d)
    d = DataFrame(
        d = d, 
        dstand = d / cutoff, 
        pval = pval(e_cdf, d), 
        sd2 = sd2.d,
        od2 = od2.d,
        sd2mu = sd2.d / sd2.mu,
        od2mu = od2.d / od2.mu,
        gh = sd2.d / nlv
        )
    ## Coefs for graphic SD2/mu - OD2/mu
    a = 1 / od2.nu * cutoff 
    b = -sd2.nu / od2.nu
    coefs = [a; b]
    ## 
    Occdds(d, fitm, e_cdf, nu, cutoff, sd2, od2, coefs, par) 
end

"""
    predict(object::Occdds, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occdds, X)
    nlv = nco(object.fitm.T)
    tscales = object.sd2.tscales    
    ## SD
    T = transf(object.fitm, X)
    Q = eltype(T)
    m, nlv = size(T)
    fscale!(T, tscales)
    sd2 = vec(eucl2(T, zeros(Q, nlv)'))
    ## OD
    E = xresid(object.fitm, X)
    od2 = rownorm2(E)
    ## Consensus
    d = object.sd2.nu / object.sd2.mu * sd2 + object.od2.nu / object.od2.mu * od2
    ## End
    d = DataFrame(
        d = d, 
        dstand = d / object.cutoff, 
        pval = pval(object.e_cdf, d),
        sd2 = sd2,
        od2 = od2, 
        sd2mu = sd2 / object.sd2.mu,
        od2mu = od2 / object.od2.mu,
        gh = sd2 / nlv
        )
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i in eachindex(d.d)]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

