"""
    occdds(; kwargs...)
    occdds(object, X; kwargs...)
One-class classification (OCC) using DD-Simca.
* `fitm` : The preliminary model (e.g., object `fitm` returned by functions `pcasvd` or `plskern`) 
    that was fitted on the training data assumed to represent the reference (= target) class.
* `X` : Training X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to consider. By default, it is the maximum nb. of LVs
    defined in model `object`.
* `fcentr` : A function that computes the centers of the empirical distributions of the squared score and orthogonal 
    distances (SD^2 and OD^2). By default, `fcentr = meanv`.
* `fscal` : A function that computes the scale (dispersion) of the empirical distributions of SD^2 and OD^2. 
     By default, `fscal = stdv`.
* `alpha` : Risk-I level to compute the parametric quantile (re-scaled Chi-square) of the consensus variable.

The function implements OCC based on the outlierness `d` as defined in the so called "DD-Simca method" (see related 
references). The principle is described below.

In DD-Simca, SD^2 and OD^2 computed on the training set are assumed to have approximately independent re-scaled 
Chi-square distributions, as follows:
* SD^2 ~ g1 * Chi-square(nu1)
* OD^2 ~ g2 * Chi-square(nu2)
where {nu1, nu2} are degrees of freedom (dofs) and {g1, g2} scaling parameters of the respective Chi-square 
distributions. 

This is equivalent to assume that the scaled SD^2 and OD^2 have the following Chi-square distributions:
* (1 / g1) * SD^2 ~ Chi-square(nu1)
* (1 / g2) * OD^2 ~ Chi-square(nu2)
or by using the properties of the Chi-square distribution (see in **Details** below):
* (nu1 / mu1) * SD^2 ~ Chi-square(nu1)
* (nu2 / mu2) * OD^2 ~ Chi-square(nu2)
where parameters {mu1, mu2} represent centers of the SD^2 and OD^2 training distributions, respectively.

Then, outlierness `d` of a given observation is finally defined by the following consensus: 
* `d` = (1 / g1) * SD^2 + (1 / g2) * OD^2
or, equivalently, by:
* `d` = (nu1 / mu1) * SD^2 + (nu2 / mu2) * OD^2

Outlier `d` is assumed to approximately follow (for the training set) a Chi-square distribution with 
nu = nu1 + nu2 dofs (assuming independance between the SD^2 and OD^2 distributions). This distribution 
is used to compute a parametric cutoff for `d` for a given risk-I level `alpha`. 
Striclty speking, this cutoff should only be applied to the training observations but, in practice, it 
is also used to classify the new observations.

Parameters {mu1, mu2} and {nu1, nu2} are estimated by the moments method on the training set. 

**Details:**

Let us note Z to represent either SD^2 or OD^2. The method assumes that Z ~ g * Chi-square(nu) or, equivalently, 
(1 / g) * Z ~ Chi-square(nu). If mu and sigma^2 represent the expectation and variance of Z (i.e., mu = E[Z] 
and sigma^2 = Var[Z]), it follows from the properties of the Chi-square distribution that:
* g = mu / nu = sigma^2 / (2 * mu)
* nu = 2 * (mu / sigma)^2 

On a paper focusing on OD^2, Nomikos & MacGregor (1995) proposed to estimate parameters {mu, sigma^2} 
(and therefore {g, nu}) by the moments method. This consists to estimate {mu, sigma^2} by the sample mean 
(or other center statistic) and variance (or other scale statistic), respectively, of the observed (training) 
distribution of Z. In DD-Simca, the same approach is applied to both SD^2 and OD^2. This allows to 
easily compute an assumed Chi-square distribution for a consensus between SD^2 and OD^2 (outlierness `d`).

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
## "EHH" = Training reference class (= target = 'in')
s = yclatrain .== "EHH"
Xref = Xtrain[s, :]    
nref = nro(Xref)
## New reference observations ("EHH") to be predicted ==> should be predicted 'in'
s = yclatest .== "EHH"
Xnew_ref = Xtest[s, :] 
nnew_ref = nro(Xnew_ref)
## New observations 'out' ("PEE") to be predicted ==> should be predicted 'out'
s = yclatest .== "PEE"
Xnew_out = Xtest[s, :] 
nnew_out = nro(Xnew_out)

## Only used to compute classification error rates
ntot = nref + nnew_ref + nnew_out
(ntot = ntot, nref, nnew_ref, nnew_out)
yref = fill("in", nref)
ynew_ref = fill("in", nnew_ref)
ynew_out = fill("in", nnew_out)

#### Fit a preliminary Pca model on the training reference data
nlv = 15
model0 = pcasvd(; nlv) 
#model0 = pcaout(; nlv) 
fit!(model0, Xref) 
fitm0 = model0.fitm ;
res = summary(model0, Xref).explvarx 
plotgrid(res.nlv, res.pvar; step = 2, xlabel = "Nb. LVs", ylabel = "% Variance explained").f
Tref = fitm0.T

#### To describe the data, 
#### project the test observations in the fitted score space
Tnew_ref = transf(model0, Xnew_ref)
Tnew_out = transf(model0, Xnew_out)
#GLMakie.activate!()   # requires GLMakie
T = vcat(Tref, Tnew_ref, Tnew_out)
group = vcat(fill("1-Train (ref)", nref), fill("2-New_ref", nnew_ref), fill("3-New_out", nnew_out))
lev = mlev(group)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
i = 1
plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], group; color, leg_title = "Type of obs.", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1), zlabel = string("PC", i + 2)).f

#### Fit the Occ model based on the fitted score space 
model = occdds()
#model = occdds(nlv = 5)
fit!(model, fitm0, Xref)
@names model 
fitm = model.fitm ;
@names fitm 
@head dref = fitm.d
cutoff = fitm.cutoff

d = dref.d
s = d .> cutoff
tsp = .4 ; color = (:orange, tsp)
f, ax = plotxy(1:nref, d; color, size = (500, 300), title = "Train (reference class)",  
    xlabel = "Observation index", ylabel = "Outlierness")
hlines!(ax, cutoff; color = :grey, linestyle = :dot, label = "Cutoff")
scatter!(ax, (1:nref)[s], d[s]; color = color[1], label = "Extreme")
f[1, 2] = Legend(f, ax, ""; framevisible = false)
f

f = Figure(size = (450, 300)) 
ax = Axis(f[1, 1]; xticks = ([1], ["Train"]), xlabel = "", ylabel = "Outlierness") 
rainclouds!(ax, fill(cutoff, nref), d; clouds = hist, jitter_width = .1, color, markersize = 10)
hlines!(ax, cutoff; color = :grey, linestyle = :dash, label = "Cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

d = dref.d
sd2mu = dref.sd2mu
od2mu = dref.od2mu
a = fitm.coefs[1]
b = fitm.coefs[2]
s = d .> cutoff
tsp = .4 ; color = (:orange, tsp)
f, ax = plotxy(sd2mu, od2mu; color, title = "Train (reference class)", xlabel = "SD2 / mu", 
    ylabel = "OD2 / mu")
scatter!(ax, sd2mu[s], od2mu[s]; color = color[1], label = "Extreme")
ablines!(ax, a, b; color = :grey, linewidth = .7, linestyle = :dash, label = "Cutoff")
f[1, 2] = Legend(f, ax, ""; framevisible = false)
f

#### Predict the new reference observations
res = predict(model, Xnew_ref) ;
@names res
@head pred = res.pred
@head dnew_ref = res.d
tab(pred)
errp(pred, ynew_ref)
conf(pred, ynew_ref).cnt

#### Predict the new observations 'out'
res = predict(model, Xnew_out) ;
@names res
@head pred = res.pred
@head dnew_out = res.d
tab(pred)
errp(pred, ynew_out)
conf(pred, ynew_out).cnt

d = vcat(dref.d, dnew_ref.d, dnew_out.d)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
f, ax = plotxy(1:length(d), d, group; color, size = (500, 300), leg = false, 
    xlabel = "Observation index", ylabel = "Outlierness")
hlines!(ax, cutoff; color = :grey, linestyle = :dot, label = "Cutoff")
f[1, 2] = Legend(f, ax, "Type of obs."; framevisible = false)
f

d = vcat(dref.d, dnew_ref.d, dnew_out.d)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
groupnum = vcat(fill(1, nref), fill(2, nnew_ref), fill(3, nnew_out))
cols = vcat(fill(color[1], nref), fill(color[2], nnew_ref), fill(color[3], nnew_out))
CairoMakie.activate!()
f = Figure(size = (600, 300))
ax = Axis(f[1, 1]; xticks = (1:3, lev), xlabel = "", ylabel = "Outlierness") 
rainclouds!(ax, groupnum, d; clouds = hist, jitter_width = .1, color = cols, markersize = 10)
hlines!(ax, cutoff; color = :grey, linestyle = :dash, linewidth = 1, label = "cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

d = dref.d
sd2mu = dref.sd2mu
od2mu = dref.od2mu
a = fitm.coefs[1]
b = fitm.coefs[2]
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
f, ax = plotxy(sd2mu, od2mu; size = (600, 300), color = color[1], xlabel = "SD2 / mu", 
    ylabel = "OD2 / mu", label = "1-Train (ref)")
scatter!(ax, dnew_ref.sd2mu, dnew_ref.od2mu; color = color[2], label = "2-New_ref")
scatter!(ax, dnew_out.sd2mu, dnew_out.od2mu; color = color[3], label = "3-New_out")
ablines!(ax, a, b; color = :grey, linewidth = .7, linestyle = :dash, label = "Cutoff")
f[1, 2] = Legend(f, ax, "Type of obs."; framevisible = false)
f
```
""" 
occdds(; kwargs...) = JchemoModel(occdds, nothing, kwargs)

function occdds(fitm, X; kwargs...) 
    X = ensure_mat(X)
    Q = eltype(X)
    par = recovkw(ParOccdds{Q}, kwargs).par 
    @assert 0 <= par.alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."    
    if isnothing(par.nlv)
        par.nlv = nco(fitm.T)
    else
        par.nlv = min(par.nlv, nco(fitm.T))
    end
    sd = outsd(fitm; par.nlv)
    od = outod(fitm, X; par.nlv)
    ## Estimates for SD^2
    d = sd.d.^2 
    mu = par.fcentr(d)
    sigma = par.fscal(d)
    g = sigma^2 / (2 * mu)
    nu = 2 * (mu / sigma)^2
    nu = max(1, round(Int, nu))
    cutoff = mu / nu * quantile(Chisq(nu), 1 - par.alpha)
    sd2 = (d = d, mu, sigma, g, nu, cutoff, tscales = sd.tscales)
    ## Estimates for OD^2
    d = od.d.^2 
    mu = par.fcentr(d)
    sigma = par.fscal(d)
    g = sigma^2 / (2 * mu)
    nu = 2 * (mu / sigma)^2
    nu = max(1, round(Int, nu))
    cutoff = mu / nu * quantile(Chisq(nu), 1 - par.alpha)
    od2 = (d = d, mu, sigma, g, nu, cutoff)
    ## Consensus
    nu = sd2.nu + od2.nu
    cutoff = quantile(Chisq(nu), 1 - par.alpha)
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
        gh = sd2.d / par.nlv
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
    nlv = object.par.nlv
    tscales = object.sd2.tscales    
    ## SD^2
    T = transf(object.fitm, X, nlv)
    Q = eltype(T)
    m = nro(T)
    fscale!(T, tscales)
    sd2 = vec(eucl2(T, zeros(Q, 1, nlv)))
    ## OD^2
    E = xresid(object.fitm, X, nlv)
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

