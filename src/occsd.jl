"""
    occsd(; kwargs...)
    occsd(fitm; kwargs...)
One-class classification (OCC) using PCA/PLS score distance (SD).
* `fitm` : The preliminary model (e.g., object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the reference (= target) class.
Keyword arguments:
* `typcut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.

OCC using outlierness `d` as defined in function `outsd`. 

If a new observation has d higher than a given `cutoff`, the observation is assumed to not belong to the training 
(= reference = target) class. The `cutoff` is computed with non-parametric heuristics, as follows. Noting `d` the vector 
of outliernesses computed on the training class:
* If `typcut` = `:mad`, then `cutoff` = MED(`d`) + `cri` * MAD(`d`). 
* If `typcut` = `:q`, then `cutoff` is computed by the empirical quantile of `d` for risk-I = `alpha`.
Approximate parametric cutoffs have been proposed in the literature (e.g., Nomikos & MacGregor 1995, Hubert et al. 2005,
Pomerantsev 2008). Whatever the approximation method used, it is recommended to tune the cutoff depending on the 
detection objectives. 

Details on outputs:
* `d` : Outlierness.
* `dstand` : Standardized outlierness defined by `d` / `cutoff`.
* `pval` : Empirical Prob(`d` > `cutoff`).
* `gh` : Indicator 'GH' provided in the software referred to as 'Winisi', computed as GH = SD^2 / nlv, where nlv is 
    the nb. scores used in the dimension reduction model. Winisi considers that GH > 3 is extreme.

## References
M. Hubert, V. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust principal components analysis. 
Technometrics, 47, 64-79.

Nomikos, V., MacGregor, J.F., 1995. Multivariate SPC Charts for Monitoring Batch Processes. null 37, 41-59. 
https://doi.org/10.1080/00401706.1995.10485888

Pomerantsev, A.L., 2008. Acceptance areas for multivariate classification derived by projection methods. 
Journal of Chemometrics 22, 601-609. https://doi.org/10.1002/cem.1147

K. Vanden Branden, M. Hubert (2005). Robust classification in high dimension based on the SIMCA method. 
Chem. Lab. Int. Syst, 79, 10-21.

K. Varmuza, V. Filzmoser (2009). Introduction to multivariate statistical analysis in chemometrics. 
CRC Press, Boca Raton.

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
## The training reference class (= target = 'in') is "EHH" 
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

#### Project the test observations in the fitted score space 'in'
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
model = occsd(cri = 2.5)
#model = occsd(typcut = :mad, cri = 4)
#model = occsd(typcut = :q, alpha = .01)
fit!(model, fitm0) 
@names model 
fitm = model.fitm ;
@names fitm 
@head dtrain_in = fitm.d
fitm.cutoff

d = dtrain_in.dstand
f, ax = plotxy(1:length(d), d; color = (:red, .3), size = (500, 300), xlabel = "Observation index", 
    ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
s = d .> 1
scatter!(ax, (1:length(d))[s], d[s]; color = :red)
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
    title = "SD", xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
"""
occsd(; kwargs...) = JchemoModel(occsd, nothing, kwargs)

function occsd(fitm; kwargs...)
    par = recovkw(ParOcc, kwargs).par
    @assert in(par.typcut, [:mad, :q]) "Argument 'typcut' must be :mad or :q."
    @assert 0 <= par.alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."
    res = outsd(fitm)
    d = res.d
    tscales = res.tscales
    nlv = nco(fitm.T)
    if par.typcut == :mad
        cutoff = median(d) + par.cri * madv(d)
    elseif par.typcut == :q
        cutoff = quantile(d, 1 - par.alpha)
    end
    e_cdf = StatsBase.ecdf(d)
    d = DataFrame(
        d = d, 
        dstand = d / cutoff, 
        pval = pval(e_cdf, d), 
        gh = d.^2 / nlv
        )
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
    ## Mahalanobis distance to center (zero)
    fscale!(T, object.tscales)
    d2 = vec(eucl2(T, zeros(Q, nlv)'))
    d = sqrt.(d2)
    ## End
    d = DataFrame(
        d = d, 
        dstand = d / object.cutoff, 
        pval = pval(object.e_cdf, d), 
        gh = d2 / nlv
        )
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i in eachindex(d.d)]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

