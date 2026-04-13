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
ytrain_in = repeat(["in"], ntrain_in)
ytest_in = repeat(["in"], ntest_in)
ytest_out = repeat(["out"], ntest_out)

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
group = vcat(repeat(["Train_in"], ntrain_in), repeat(["Test_in"], ntest_in), repeat(["Test_out"], ntest_out))
color = [:purple, (:green, .7), (:red, .3)]
i = 1
plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], group; color = color, leg_title = "Type of obs.", 
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
f, ax = plotxy(1:length(d), d, group; color = color, size = (500, 300), leg_title = "Type of obs.", 
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
    par = recovkw(ParOccdds, kwargs).par 
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

