"""
    occsd(; kwargs...)
    occsd(fitm; kwargs...)
One-class classification (OCC) using PCA/PLS score distance (SD).
* `fitm` : The preliminary model (e.g., object `fitm` returned by functions `pcasvd` or `plskern`) 
    that was fitted on the training data assumed to represent the reference (= target) class.
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to consider. By default, it is the maximum nb. of LVs
    defined in model `object`.
* `typcut` : Type of cutoff. Possible values are: `:std`, `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:std` or `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.

OCC using outlierness `d` as defined in function `outsd`. 

If a new observation has d higher than a given `cutoff`, the observation is assumed to not belong to the training 
(= reference = target) class. The `cutoff` is computed with non-parametric heuristics, as follows. Noting `d` the vector 
of outliernesses computed on the training class:
* If `typcut` = `:std`, then `cutoff` = MEAN(`d`) + `cri` * STD(`d`). 
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
yref = repeat(["in"], nref)
ynew_ref = repeat(["in"], nnew_ref)
ynew_out = repeat(["out"], nnew_out)

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
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
i = 1
plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], group; color, leg_title = "Type of obs.", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1), zlabel = string("PC", i + 2)).f

#### Fit the Occ model based on the fitted score space 
model = occsd(cri = 2.5)
#model = occsd(typcut = :std, cri = 2.5)
#model = occsd(typcut = :q, alpha = .01)
#model = occsd(nlv = 5, cri = 2.5)
fit!(model, fitm0) 
@names model 
fitm = model.fitm ;
@names fitm 
@head dref = fitm.d
fitm.cutoff

d = dref.dstand
s = d .> 1
tsp = .4 ; color = (:orange, tsp)
f, ax = plotxy(1:length(d), d; color, size = (500, 300), title = "Train (reference class)",  
    xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; color = :grey, linestyle = :dot)
scatter!(ax, (1:length(d))[s], d[s]; color = color[1], label = "Extreme")
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

d = vcat(dref.dstand, dnew_ref.dstand, dnew_out.dstand)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
f, ax = plotxy(1:length(d), d, group; color, size = (500, 300), leg_title = "Type of obs.", 
    title = "SD", xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; color = :grey, linestyle = :dot)
f
```
"""
occsd(; kwargs...) = JchemoModel(occsd, nothing, kwargs)

function occsd(fitm; kwargs...)
    Q = eltype(fitm.T)
    par = recovkw(ParOcc{Q}, kwargs).par
    @assert in(par.typcut, [:std, :mad, :q]) "Argument 'typcut' must be :std, :mad or :q."
    @assert 0 <= par.alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."
    if isnothing(par.nlv)
        par.nlv = nco(fitm.T)
    else
        par.nlv = min(par.nlv, nco(fitm.T))
    end
    res = outsd(fitm; par.nlv)
    d = res.d
    tscales = res.tscales
    if par.typcut == :std
        cutoff = meanv(d) + par.cri * stdv(d)    
    elseif par.typcut == :mad
        cutoff = medv(d) + par.cri * madv(d)
    elseif par.typcut == :q
        cutoff = quantv(d, 1 - par.alpha)
    end
    e_cdf = StatsBase.ecdf(d)
    d = DataFrame(
        d = d, 
        dstand = d / cutoff, 
        pval = pval(e_cdf, d), 
        gh = d.^2 / par.nlv
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
    nlv = object.par.nlv
    T = transf(object.fitm, X, nlv)
    m = nro(T)
    Q = eltype(T)
    ## Mahalanobis distance to center (zero)
    fscale!(T, object.tscales)
    d2 = vec(eucl2(T, zeros(Q, 1, nlv)))
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

