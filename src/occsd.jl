"""
    occsd(; kwargs...)
    occsd(fitm; kwargs...)
One-class classification using PCA/PLS score distance (SD).
* `fitm` : The preliminary model (e.g. object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the reference class.
Keyword arguments:
* `cut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `cut` = `:mad`, a constant. See thereafter.
* `risk` : When `cut` = `:q`, a risk-I level. See thereafter.

In this method, the outlierness `d` of an observation is defined by its score distance (SD), ie. the Mahalanobis 
distance between the projection of the observation on the score plan defined by the fitted (e.g. PCA) model and the 
"center" (always defined by zero) of the score plan.

If a new observation has `d` higher than a given `cutoff`, the observation is assumed to not belong to the training 
(= reference) class. The `cutoff` is computed with non-parametric heuristics. Noting [d] the vector of outliernesses 
computed on the training class:
* If `cut` = `:mad`, then `cutoff` = MED([d]) + `cri` * MAD([d]). 
* If `cut` = `:q`, then `cutoff` is estimated from the empirical cumulative density function 
    computed on [d], for a given risk-I (`risk`).
Alternative approximate cutoffs have been proposed in the literature (e.g.: Nomikos & MacGregor 1995, Hubert et al. 2005,
Pomerantsev 2008). Typically, and whatever the approximation method used to compute the cutoff, it is recommended to tune 
this cutoff depending on the detection objectives. 

**Outputs**
* `pval`: Estimate of p-value (see functions `pval`) computed from the training distribution [d]. 
* `dstand`: standardized distance defined as `d` / `cutoff`. A value `dstand` > 1 may be considered as extreme 
    compared to the distribution of the training data.  
* `gh` is the Winisi "GH" (usually, GH > 3 is considered as extreme).
Specific for function `predict`:
* `pred`: class prediction
    * `dstand` <= 1 ==> `in`: the observation is expected to belong to the training class, 
    * `dstand` > 1  ==> `out`: extreme value, possibly not belonging to the same class as the training. 

## References
M. Hubert, V. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust principal components 
analysis. Technometrics, 47, 64-79.

Nomikos, V., MacGregor, J.F., 1995. Multivariate SPC Charts for Monitoring Batch Processes. null 37, 41-59. 
https://doi.org/10.1080/00401706.1995.10485888

Pomerantsev, A.L., 2008. Acceptance areas for multivariate classification derived by projection methods. 
Journal of Chemometrics 22, 601-609. https://doi.org/10.1002/cem.1147

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
Xtest = Xp[s, :]
Ytest = Y[s, :]

## Build the example data
## - cla_train is the reference class (= 'in'), "EHH" 
cla_train = "EHH"
s = Ytrain.typ .== cla_train
Xtrain_fin = Xtrain[s, :]    
ntrain = nro(Xtrain_fin)
## cla_test contains the observations to be predicted (i.e. to be 'in' or 'out' of cla_train), 
## a mix of "EEH" and "PEE" 
cla_test1 = "EHH"   # should be predicted 'in'
s = Ytest.typ .== cla_test1
Xtest_fin1 = Xtest[s, :] 
ntest1 = nro(Xtest_fin1)
##
cla_test2 = "PEE"   # should be predicted 'out'
s = Ytest.typ .== cla_test2
Xtest_fin2 = Xtest[s, :] 
ntest2 = nro(Xtest_fin2)
##
Xtest_fin = vcat(Xtest_fin1, Xtest_fin2)
## Only used to compute error rates
ytrain_fin = repeat(["in"], ntrain)
ytest_fin = [repeat(["in"], ntest1); repeat(["out"], ntest2)]
y_fin = vcat(ytrain_fin, ytest_fin)
## 
ntot = ntrain + ntest1 + ntest2
(ntot = ntot, ntrain, ntest1, ntest2)

#### Preliminary PCA fitted model
nlv = 15
model0 = pcasvd(; nlv) 
#model0 = pcaout(; nlv) 
fit!(model0, Xtrain_fin) 
res = summary(model0, Xtrain_fin).explvarx 
plotgrid(res.nlv, res.pvar; step = 2, xlabel = "Nb. LVs", ylabel = "% Variance explained").f
Ttrain = model0.fitm.T
Ttest = transf(model0, Xtest_fin)
T = vcat(Ttrain, Ttest)
i = 1
group = vcat(repeat(["Train-EHH"], ntrain), repeat(["Test-EHH"], ntest1), repeat(["Test-PEE"], ntest2))
color = [:red, :blue, (:green, .5)]
plotxy(T[:, i], T[:, i + 1], group; color = color, leg_title = "Type of obs.", xlabel = string("PC", i), 
    ylabel = string("PC", i + 1)).f

#### Occ
## Training
model = occsd(; cri = 2.5)
#model = occsd(cut = :mad, cri = 4)
#model = occsd(cut = :q, risk = .01)
fit!(model, model0.fitm) 
@names model 
fitm = model.fitm ;
@names fitm 
@head dtrain = fitm.d
fitm.cutoff
d = dtrain.dstand
f, ax = plotxy(1:length(d), d; color = (:green, .5), size = (500, 300), xlabel = "Obs. index", 
    ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
## Prediction of Test
res = predict(model, Xtest_fin) 
@names res
@head pred = res.pred
@head dtest = res.d
tab(pred)
errp(pred, ytest_fin)
conf(pred, ytest_fin).cnt
##
d = vcat(dtrain.dstand, dtest.dstand)
color = [:red, :blue, (:green, .5)]
f, ax = plotxy(1:length(d), d, group; color = color, size = (500, 300), leg_title = "Type of obs.", 
    xlabel = "Obs. index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
"""
occsd(; kwargs...) = JchemoModel(occsd, nothing, kwargs)

function occsd(fitm; kwargs...)
    par = recovkw(ParOcc, kwargs).par
    @assert in(par.cut, [:mad, :q]) "Argument 'cut' must be :mad or :q."
    @assert 0 <= par.risk <= 1 "Argument 'risk' must ∈ [0, 1]."
    T = copy(fitm.T) # remove side effect of fscale!
    Q = eltype(T)
    nlv = nco(T)
    ## Mahalanobis distance
    tscales = colstd(T, fitm.weights)
    fscale!(T, tscales)
    d2 = vec(euclsq(T, zeros(Q, nlv)'))   # the center is defined as 0
    d = sqrt.(d2)
    ## End
    if par.cut == :mad
        cutoff = median(d) + par.cri * madv(d)
    elseif par.cut == :q
        cutoff = quantile(d, 1 - par.risk)
    end
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
    ## Mahalanobis distance
    fscale!(T, object.tscales)
    d2 = vec(euclsq(T, zeros(Q, nlv)'))
    d = sqrt.(d2)
    ## End
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = p_val, gh = d2 / nlv)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

