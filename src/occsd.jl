"""
    occsd(; kwargs...)
    occsd(fitm; kwargs...)
One-class classification (OCC) using (PCA/PLS) score distance (SD).
* `fitm` : The preliminary model (e.g., object `fitm` returned by functions `pcasvd` or `plskern`) 
    that was fitted on the training data assumed to represent the reference (= target) class.
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to consider. By default, it is the maximum nb. of LVs
    defined in model `fitm`.
* `typcut` : Type of cutoff for outlierness `d`. Possible values are: `:std`, `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:std` or `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.

OCC using outlierness `d` as defined in function `outsd`. 

If a new observation has `d` higher than a given cutoff value (`cutoff_d`), the observation is assumed 
to not belong to the training (= reference = target) class. Value `cutoff_d` is computed with non-parametric 
heuristics, as follows. Let `d` be the vector of outliernesses computed on the *training class*:
* If `typcut` = `:std`, then `cutoff_d` = MEAN(`d`) + `cri` * STD(`d`). 
* If `typcut` = `:mad`, then `cutoff_d` = MED(`d`) + `cri` * MAD(`d`). 
* If `typcut` = `:q`, then `cutoff_d` is computed by the empirical quantile of `d` for risk-I = `alpha`.
Approximate parametric cutoffs have been proposed in the literature (e.g., Nomikos & MacGregor 1995, Hubert et al. 2005,
Pomerantsev 2008). Whatever the approximation method used, it is recommended to tune the cutoff depending on the 
detection objectives. 

Details on outputs:
* `d` : Outlierness.
* `dstand_cut` : Standardized outlierness defined by `d` / `cutoff_d`.
* `pval` : Empirical Prob(`d` > `cutoff_d`).
* `gh` : Indicator 'GH' provided in the software referred to as 'Winisi', computed as GH = SD^2 / nlv, where nlv is 
    the nb. scores used in the dimension reduction model. Winisi considers that GH > 3 is extreme.

# References
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

# Examples
```julia
using Jchemo, JchemoData, JLD2, GLMakie, CairoMakie
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

#### Build the data used in the present example
# "EHH" = Training reference class (= target = 'in')
s = yclatrain .== "EHH"
Xtrain_ref = Xtrain[s, :]    
ntrain_ref = nro(Xtrain_ref)
# New reference observations ("EHH") to be predicted ==> should be predicted 'in'
s = yclatest .== "EHH"
Xnew_ref = Xtest[s, :] 
nnew_ref = nro(Xnew_ref)
# New observations 'out' ("PEE") to be predicted ==> should be predicted 'out'
s = yclatest .== "PEE"
Xnew_out = Xtest[s, :] 
nnew_out = nro(Xnew_out)
# Required to compute classification error rates
ntot = ntrain_ref + nnew_ref + nnew_out
(ntot = ntot, ntrain_ref, nnew_ref, nnew_out)
yref = fill("in", ntrain_ref)
ynew_ref = fill("in", nnew_ref)
ynew_out = fill("out", nnew_out)

#### Preliminary data description
# Fit a preliminary Pca model on the training reference data
nlv = 15
model = pcasvd(; nlv) 
#model = pcaout(; nlv) 
fit!(model, Xtrain_ref) 
fitm = model.fitm ;
res = summary(model, Xtrain_ref).explvarx 
plotgrid(res.nlv, res.pvar; step = 2, xlabel = "Nb. LVs", ylabel = "% Variance explained").f
Ttrain_ref = fitm.T
# Project the test observations in the fitted score space
Tnew_ref = transf(model, Xnew_ref)
Tnew_out = transf(model, Xnew_out)
#GLMakie.activate!()   # requires GLMakie
T = vcat(Ttrain_ref, Tnew_ref, Tnew_out)
group = vcat(fill("1-Train_ref", ntrain_ref), fill("2-New_ref", nnew_ref), fill("3-New_out", nnew_out))
lev = mlev(group)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
i = 1
plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], group; color, leg_title = "Type of obs.", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1), zlabel = string("PC", i + 2)).f
#### End

#### Define the embedding model to consider
nlv = 10 
model = pcasvd(; nlv)
fit!(model, Xtrain_ref)
fitm0 = model.fitm ;

#### Fit the Occ model based on the embedding 
model = occsd(cri = 2.5)
#model = occsd(typcut = :std, cri = 2.5)
#model = occsd(typcut = :q, alpha = .01)
#model = occsd(nlv = 5, cri = 2.5)
fit!(model, fitm0) 
@names model 
fitm = model.fitm ;
@names fitm 
@head dtrain_ref = fitm.d
cutoff_d = fitm.cutoff_d

d = dtrain_ref.d
s = d .> cutoff_d
tsp = .4 ; color = (:orange, tsp)
f, ax = plotxy(1:ntrain_ref, d; color, size = (500, 300), title = "Train (reference class)",  
    xlabel = "Observation index", ylabel = "Outlierness")
hlines!(ax, cutoff_d; color = :grey, linestyle = :dot, label = "Cutoff")
scatter!(ax, (1:ntrain_ref)[s], d[s]; color = color[1], label = "Extreme")
f[1, 2] = Legend(f, ax, ""; framevisible = false)
f

f = Figure(size = (450, 300)) 
ax = Axis(f[1, 1]; xticks = ([0], [""]), title = "Train (reference class)", 
    xlabel = "", ylabel = "Outlierness") 
rainclouds!(ax, fill(1, ntrain_ref), d; clouds = hist, jitter_width = .1, color, markersize = 10)
hlines!(ax, cutoff_d; color = :grey, linestyle = :dash, label = "Cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
f

#### Predict the new observations 'ref'
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

d = vcat(dtrain_ref.d, dnew_ref.d, dnew_out.d)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
f, ax = plotxy(1:length(d), d, group; color, size = (500, 300), leg = false, 
    xlabel = "Observation index", ylabel = "Outlierness")
hlines!(ax, cutoff_d; color = :grey, linestyle = :dot, label = "Cutoff")
f[1, 2] = Legend(f, ax, "Type of obs."; framevisible = false)
f

d = vcat(dtrain_ref.d, dnew_ref.d, dnew_out.d)
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
groupnum = vcat(fill(1, ntrain_ref), fill(2, nnew_ref), fill(3, nnew_out))
cols = vcat(fill(color[1], ntrain_ref), fill(color[2], nnew_ref), fill(color[3], nnew_out))
f = Figure(size = (600, 300))
ax = Axis(f[1, 1]; xticks = (1:3, lev), xlabel = "", ylabel = "Outlierness") 
rainclouds!(ax, groupnum, d; clouds = hist, jitter_width = .1, color = cols, markersize = 10)
hlines!(ax, cutoff_d; color = :grey, linestyle = :dash, linewidth = 1, label = "Cutoff")
Legend(f[1, 2], ax, ""; nbanks = 1, rowgap = 10, framevisible = false)
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
    if par.typcut == :std
        cutoff_d = meanv(d) + par.cri * stdv(d)    
    elseif par.typcut == :mad
        cutoff_d = medv(d) + par.cri * madv(d)
    elseif par.typcut == :q
        cutoff_d = quantv(d, 1 - par.alpha)
    end
    e_cdf = StatsBase.ecdf(d)
    d = DataFrame(
        d = d, 
        dstand_cut = d / cutoff_d, 
        pval_d = pval(e_cdf, d), 
        gh = d.^2 / par.nlv
        )
    Occsd(d, fitm, res.tscales, e_cdf, cutoff_d, par)
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
    # Mahalanobis distance to center (zero)
    fscale!(T, object.tscales)
    d2 = vec(eucl2(T, zeros(Q, 1, nlv)))
    d = sqrt.(d2)
    # End
    d = DataFrame(
        d = d, 
        dstand_cut = d / object.cutoff_d, 
        pval_d = pval(object.e_cdf, d), 
        gh = d2 / nlv
        )
    pred = [if d.dstand_cut[i] <= 1 "in" else "out" end for i in eachindex(d.d)]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

