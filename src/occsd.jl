"""
    occsd(; kwargs...)
    occsd(fitm; kwargs...)
One-class classification using PCA/PLS score distance (SD).
* `fitm` : The preliminary model (e.g. PCA; object `fitm`) that was fitted 
    on the training data assumed to represent the training class.
Keyword arguments:
* `mcut` : Type of cutoff. Possible values are: `:mad`, `:q`. 
    See Thereafter.
* `cri` : When `mcut` = `:mad`, a constant. See thereafter.
* `risk` : When `mcut` = `:q`, a risk-I level. See thereafter.

In this method, the outlierness `d` of an observation is 
defined by its score distance (SD), ie. the Mahalanobis distance 
between the projection of the observation on the score plan 
defined by the fitted (e.g. PCA) model and the center of the 
score plan.

If a new observation has `d` higher than a given `cutoff`, the 
observation is assumed to not belong to the training (= reference) 
class. The `cutoff` is computed with non-parametric heuristics. 
Noting [d] the vector of outliernesses computed on the training class:
* If `mcut` = `:mad`, then `cutoff` = median([d]) + `cri` * madv([d]). 
* If `mcut` = `:q`, then `cutoff` is estimated from the empirical 
    cumulative density function computed on [d], for a given 
    risk-I (`risk`). 
Alternative approximate cutoffs have been proposed in the 
literature (e.g.: Nomikos & MacGregor 1995, Hubert et al. 2005,
Pomerantsev 2008). Typically, and whatever the approximation method 
used to compute the cutoff, it is recommended to tune this cutoff 
depending on the detection objectives. 

**Outputs**
* `pval`: Estimate of p-value (see functions `pval`) computed 
    from the training distribution [d]. 
* `dstand`: standardized distance defined as `d` / `cutoff`. 
    A value `dstand` > 1 may be considered as extreme compared to 
    the distribution of the training data.  
* `gh` is the Winisi "GH" (usually, GH > 3 is considered as 
    extreme).
Specific for function `predict`:
* `pred`: class prediction
    * `dstand` <= 1 ==> `in`: the observation is expected to 
        belong to the training class, 
    * `dstand` > 1  ==> `out`: extreme value, possibly not 
        belonging to the same class as the training. 

## References
M. Hubert, P. J. Rousseeuw, K. Vanden Branden (2005). 
ROBPCA: a new approach to robust principal components 
analysis. Technometrics, 47, 64-79.

Nomikos, P., MacGregor, J.F., 1995. Multivariate SPC Charts for 
Monitoring Batch Processes. null 37, 41-59. 
https://doi.org/10.1080/00401706.1995.10485888

Pomerantsev, A.L., 2008. Acceptance areas for multivariate 
classification derived by projection methods. Journal of Chemometrics 
22, 601-609. https://doi.org/10.1002/cem.1147

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/challenge2018.jld2") 
@load db dat
pnames(dat)
X = dat.X    
Y = dat.Y
model = savgol(npoint = 21, deriv = 2, degree = 3)
fit!(model, X) 
Xp = transf(model, X) 
s = Bool.(Y.test)
Xtrain = rmrow(Xp, s)
Ytrain = rmrow(Y, s)
Xtest = Xp[s, :]
Ytest = Y[s, :]

## Below, the reference class is "EEH"
cla1 = "EHH" ; cla2 = "PEE" ; cod = "out"   # here cla2 should be detected
#cla1 = "EHH" ; cla2 = "EHH" ; cod = "in"   # here cla2 should not be detected
s1 = Ytrain.typ .== cla1
s2 = Ytest.typ .== cla2
zXtrain = Xtrain[s1, :]    
zXtest = Xtest[s2, :] 
ntrain = nro(zXtrain)
ntest = nro(zXtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)
ytrain = repeat(["in"], ntrain)
ytest = repeat([cod], ntest)

## Group description
model = pcasvd(nlv = 10) 
fit!(model, zXtrain) 
Ttrain = model.fitm.T
Ttest = transf(model, zXtest)
T = vcat(Ttrain, Ttest)
group = vcat(repeat(["1"], ntrain), repeat(["2"], ntest))
i = 1
plotxy(T[:, i], T[:, i + 1], group; title_leg = "Class", xlabel = string("PC", i),  
    ylabel = string("PC", i + 1)).f

#### Occ
## Preliminary PCA fitted model
model0 = pcasvd(nlv = 30) 
fit!(model0, zXtrain)
## Outlierness
model = occsd()
#model = occsd(mcut = :mad, cri = 4)
#model = occsd(mcut = :q, risk = .01)
fit!(model, model0.fitm) 
pnames(model) 
pnames(model.fitm) 
@head d = model.fitm.d
d = d.dstand
f, ax = plotxy(1:length(d), d; size = (500, 300), xlabel = "Obs. index", 
    ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f

res = predict(model, zXtest) ;
pnames(res)
@head res.d
@head res.pred
tab(res.pred)
errp(res.pred, ytest)
conf(res.pred, ytest).cnt
d1 = model.fitm.d.dstand
d2 = res.d.dstand
d = vcat(d1, d2)
f, ax = plotxy(1:length(d), d, group; size = (500, 300), title_leg = "Class", xlabel = "Obs. index", 
    ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
"""
occsd(; kwargs...) = JchemoModel(occsd, nothing, kwargs)

function occsd(fitm; kwargs...)
    par = recovkw(ParOcc, kwargs).par 
    @assert 0 <= par.risk <= 1 "Argument 'risk' must ∈ [0, 1]."
    T = copy(fitm.T) # remove side effect of fscale!
    Q = eltype(T)
    nlv = nco(T)
    tscales = colstd(T, fitm.weights)
    fscale!(T, tscales)
    d2 = vec(euclsq(T, zeros(Q, nlv)'))
    d = sqrt.(d2)
    par.mcut == :mad ? cutoff = median(d) + par.cri * madv(d) : nothing
    par.mcut == :q ? cutoff = quantile(d, 1 - par.risk) : nothing
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val, gh = d2 / nlv)
    Occsd(d, fitm, tscales, e_cdf, cutoff, par)
end

"""
    predict(object::Occsd, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occsd, X)
    T = transf(object.fitm, X)
    Q = eltype(T)
    m, nlv = size(T)
    fscale!(T, object.tscales)
    d2 = vec(euclsq(T, zeros(Q, nlv)'))
    d = sqrt.(d2)
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = p_val, gh = d2 / nlv)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

