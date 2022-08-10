struct Occsd
    d
    fm
    Sinv::Matrix{Float64}
    e_cdf
    cutoff::Real   
    nlv::Int64
end

struct Occod
    d
    fm
    e_cdf
    cutoff::Real   
    nlv::Int64
end

"""
    occsd(object::Union{Pca, Plsr}; nlv = nothing,
        typc = "mad", alpha = .05, cri = 3)
One-class classification using PCA/PLS score distance (SD).

* `object` : The PCA/PLS fitted model.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, 
    it is the maximum nb. of components of the fitted model.
* `typc` : Type of cutoff ("mad" or "q"). See below.
* `cri` : When `typc = "mad"`, constant used for computing the 
    cutoff detecting extreme values.
* `alpha` : When `typc = "q"`, risk-I level used for computing the cutoff 
    detecting extreme values.

In this method, the "outlierness measure" `d` of a given observation
is the score distance (SD) of this observation, ie. the Mahalanobis distance
between the projection of the observation on the PCA or PLS score plan and the 
center of the score plan.

A heuristic cutoff for detecting an "extreme" outlierness is computed on the training data `X.
* If `typc = "mad"`: The cutoff is computed by median(`d`) + `cri` * mad(`d`). 
* If `typc = "q"`: The cutoff is estimated from the empirical cdf of `d`. 

Alternative methods of cutoff computation can be found in the literature 
(e.g.: Nomikos & MacGregor 1995, Hubert et al. 2005, Pomerantsev 2008).

`dstand` is the standardized distance defined as `d / cutoff`. 
A value `dstand` > 1 may be considered as extreme compared to the distribution
of the training data. `pval` is the p-value computed from 
empirical (training) cdf. `gh` is the Winisi "GH" (usually, GH > 3 is considered as "extreme").

In the output `pred` of fonction `predict`, an observation is classified 
as 0 (i.e. belonging to the training class) when `dstand` <= 1 and 
1 (i.e. extreme) when `dstand` > 1. 

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
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "challenge2018.jld2") 
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
zXtrain = Xtrain[s1, :]  
zXtest = Xtest[s2, :] 
ntrain = nro(zXtrain)
ntest = nro(zXtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

fm = pcasvd(zXtrain, nlv = 5) ; 
Ttrain = fm.T
Ttest = transform(fm, zXtest)
T = vcat(Ttrain, Ttest)
group = vcat(repeat(["0-Train"], ntrain), repeat(["1-Test"], ntest))
i = 1
plotxy(T[:, i], T[:, i + 1], group;
    xlabel = string("PC", i), ylabel = string("PC", i + 1)).f

#### End data

nlv = 10
fm0 = pcasvd(zXtrain; nlv = nlv) ;

fm = occsd(fm0) ;
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
f, ax = plotxy(1:ntot, d, group)
hlines!(ax, 1)
f
```
""" 
function occsd(object::Union{Pca, Plsr}; nlv = nothing,
        typc = "mad", alpha = .05, cri = 3)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    T = @view(object.T[:, 1:nlv])
    S = Statistics.cov(T, corrected = false)
    LinearAlgebra.inv!(cholesky!(S))   # ==> S := Sinv
    d2 = vec(mahsq(T, zeros(nlv)', S))
    d = sqrt.(d2)
    typc == "mad" ? cutoff = median(d) + cri * mad(d) : nothing
    typc == "q" ? cutoff = quantile(d, 1 - alpha) : nothing
    e_cdf = StatsBase.ecdf(d)
    pval = 1 .- e_cdf(d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = pval, gh = d2 / nlv)
    Occsd(d, object, S, e_cdf, cutoff, nlv)
end

"""
    predict(object::Occsd, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occsd, X)
    nlv = object.nlv
    T = transform(object.fm, X; nlv = nlv)
    m = nro(T)
    d2 = vec(mahsq(T, zeros(nlv)', object.Sinv))
    d = sqrt.(d2)
    pval = 1 .- object.e_cdf(d)
    d = DataFrame((d = d, dstand = d / object.cutoff, pval = pval, gh = d2 / nlv))
    pred = reshape(Int64.(d.dstand .> 1), m, 1)
    (pred = pred, d)
end

"""
    occod(object::Union{Pca, Plsr}, X; nlv = nothing, 
        typc = "mad", cri = 3, alpha = .05)
One-class classification using PCA/PLS orthognal distance (OD).

* `object` : The fitted model.
* `X` : X-data (training) that were used to fit the model.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, it is the maximum 
    nb. of components.
* `typc` : Type of cutoff ("mad" or "q"). See below.
* `cri` : When `typc = "mad"`, constant used for computing the 
    cutoff detecting extreme values.
* `alpha` : When `typc = "q"`, risk-I level used for computing the cutoff 
    detecting extreme values.

In this method, the "outlierness measure" `d` of a given observation
is the orthogonal distance (OD =  "X-residuals") of this observation, ie.
the Euclidean distance between the observation and its projection to the PCA or 
PLS score plan (e.g. Hubert et al. 2005, Van Branden & Hubert 2005, p. 66; 
Varmuza & Filzmoser, 2009, p. 79).

See `?occsd` for the cutoff computation, and for numerical examples.

## References
M. Hubert, P. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust principal 
components analysis. Technometrics, 47, 64-79.

K. Vanden Branden, M. Hubert (2005). Robuts classification in high dimension based on the SIMCA method. 
Chem. Lab. Int. Syst, 79, 10-21.

K. Varmuza, P. Filzmoser (2009). Introduction to multivariate statistical analysis in chemometrics. 
CRC Press, Boca Raton.
""" 
function occod(object::Union{Pca, Plsr}, X; nlv = nothing, 
        typc = "mad", cri = 3, alpha = .05)
    X = ensure_mat(X)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    E = xresid(object, X; nlv = nlv)
    d2 = vec(sum(E .* E, dims = 2))
    d = sqrt.(d2)
    typc == "mad" ? cutoff = median(d) + cri * mad(d) : nothing
    typc == "q" ? cutoff = quantile(d, 1 - alpha) : nothing
    e_cdf = StatsBase.ecdf(d)
    pval = 1 .- e_cdf(d)
    d = DataFrame((d = d, dstand = d / cutoff, pval = pval))
    Occod(d, object, e_cdf, cutoff, nlv)
end

"""
    predict(object::Occod, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occod, X)
    E = xresid(object.fm, X; nlv = object.nlv)
    m = nro(E)
    d2 = vec(sum(E .* E, dims = 2))
    d = sqrt.(d2)
    pval = 1 .- object.e_cdf(d)
    d = DataFrame((d = d, dstand = d / object.cutoff, pval = pval))
    pred = reshape(Int64.(d.dstand .> 1), m, 1)
    (pred = pred, d)
end


