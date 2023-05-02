struct Occsd_1
    d
    fm
    Sinv::Matrix{Float64}
    fm_kde
    cutoff::Real   
    nlv::Int64
end

"""
    occsd(object::Union{Pca, Plsr}; nlv = nothing,
        typc = "mad", cri = 3, alpha = .05, kwargs...)
One-class classification using PCA/PLS score distance (SD).

* `object` : The model (e.g. PCA) that was fitted on the training data,
    assumed to represent the target class.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, 
    it is the maximum nb. of components of the fitted model.
* `typc` : Type of cutoff ("mad" or "q"). See Thereafter.
* `cri` : When `typc = "mad"`, a constant. See thereafter.
* `alpha` : When `typc = "q"`, a risk-I level. See thereafter.

In this method, the outlierness `d` is defined by its score distance (SD), 
ie. the Mahalanobis distance between the projection of the observation on the score 
plan defined by the fitted (e.g. PCA) model and the center of 
the score plan.

If a new observation has `d` higher than a given `cutoff`, the observation 
is assumed to not belong to the target class. The `cutoff` is computed 
with non-parametric heuristics:
* If `typc = "mad"`: `cutoff` = median(`d`) + `cri` * mad(`d`). 
* If `typc = "q"`: `cutoff` is estimated from a univariate gaussian 
    KDE of the distribution of `d` (see function `kde1`), for a given risk-I (`alpha`). 
Alternative approximate cutoffs have been proposed in the literature 
(e.g.: Nomikos & MacGregor 1995, Hubert et al. 2005, Pomerantsev 2008).

**Outputs**
* `pval`: KDE p-value estimate (see functions `kde1` and `pval`), 
    provided for each data observation. 
* `dstand`: standardized distance defined as `d` / `cutoff`. 
    A value `dstand` > 1 may be considered as extreme compared to the distribution
    of the training data.  Output `gh` is the Winisi "GH" (usually, GH > 3 is 
    considered as "extreme").
* `pred` (fonction `predict`): class prediction
    * `dstand` <= 1 ==> `0`: the observation is expected to belong 
        the target class, 
    * `dstand` > 1  ==> `1`: extreme value, possibly outside of the target class. 

## References
M. Hubert, P. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust 
principal components analysis. Technometrics, 47, 64-79.

Nomikos, P., MacGregor, J.F., 1995. Multivariate SPC Charts for Monitoring Batch Processes. 
null 37, 41-59. https://doi.org/10.1080/00401706.1995.10485888

Pomerantsev, A.L., 2008. Acceptance areas for multivariate classification derived by 
projection methods. Journal of Chemometrics 22, 601-609. https://doi.org/10.1002/cem.1147

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/challenge2018.jld2") 
@load db dat
X = dat.X    
Y = dat.Y
f = 21 ; pol = 3 ; d = 2 ;
Xp = savgol(snv(X); f = f, pol = pol, d = d) 
s = Bool.(Y.test)
Xtrain = rmrow(Xp, s)
Ytrain = rmrow(Y, s)
Xtest = Xp[s, :]
Ytest = Y[s, :]

g1 = "EHH" ; g2 = "PEE"
#g1 = "EHH" ; g2 = g1
s1 = Ytrain.typ .== g1
s2 = Ytest.typ .== g2
zXtrain = Xtrain[s1, :]    # = Target
zXtest = Xtest[s2, :] 
ntrain = nro(zXtrain)
ntest = nro(zXtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

fm = pcasvd(zXtrain, nlv = 5) ; 
Ttrain = fm.T
Ttest = Jchemo.transform(fm, zXtest)
T = vcat(Ttrain, Ttest)
group = vcat(repeat(["0-Train"], ntrain), repeat(["1-Test"], ntest))
i = 1
plotxy(T[:, i:(i + 1)], group;
    xlabel = string("PC", i), ylabel = string("PC", i + 1)).f

#### End data

nlv = 10
fm0 = pcasvd(zXtrain; nlv = nlv) ;

fm = occsd(fm0) ;
#fm = occsd(fm0; typc = "q", alpha = .025) ;
#fm = occod(fm0, zXtrain) ;
#fm = occsdod(fm0, zXtrain) ;
fm.d
hist(fm.d.dstand; bins = 50)

res = Jchemo.predict(fm, zXtest) ;
res.d
res.pred
tab(res.pred)

d1 = fm.d.dstand
d2 = res.d.dstand
d = vcat(d1, d2)
f, ax = plotxy(1:length(d), d; 
    resolution = (500, 400), xlabel = "Obs. index", 
    ylabel = "Standardized distance")
hlines!(ax, 1)
f
```
""" 
function occsd(object::Union{Pca, Plsr}; nlv = nothing,
        typc = "mad", alpha = .05, cri = 3, kwargs...)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    T = @view(object.T[:, 1:nlv])
    S = Statistics.cov(T, corrected = false)
    LinearAlgebra.inv!(cholesky!(S))   # ==> S := Sinv
    d2 = vec(mahsq(T, zeros(nlv)', S))
    d = sqrt.(d2)
    typc == "mad" ? cutoff = median(d) + cri * mad(d) : nothing
    typc == "q" ? cutoff = quantile(d, 1 - alpha) : nothing
    fm_kde = kde1(d; kwargs...)
    p_val = pval(fm_kde, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val, 
        gh = d2 / nlv)
    Occsd_1(d, object, S, fm_kde, cutoff, nlv)
end

"""
    predict(object::Occsd_1, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occsd_1, X)
    nlv = object.nlv
    T = transform(object.fm, X; nlv = nlv)
    m = nro(T)
    d2 = vec(mahsq(T, zeros(nlv)', object.Sinv))
    d = sqrt.(d2)
    p_val = pval(object.fm_kde, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, 
        pval = p_val, gh = d2 / nlv)
    pred = reshape(Int64.(d.dstand .> 1), m, 1)
    (pred = pred, d)
end

