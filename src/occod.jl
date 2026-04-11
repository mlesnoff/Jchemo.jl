"""
    occod(; kwargs...)
    occod(fitm, X; kwargs...)
One-class classification (OCC) using PCA/PLS orthognal distance (OD).
* `fitm` : The preliminary model (e.g., object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the reference (= target) class.
* `X` : Training X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `typcut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.

OCC using outlierness `d` as defined in function `outod`.

If a new observation has d higher than a given `cutoff`, the observation is assumed to not belong to the training 
(= reference = target) class. The `cutoff` is computed with non-parametric heuristics, as follows. Noting `d` the vector 
of outliernesses computed on the training class:
* If `typcut` = `:mad`, then `cutoff` = MED(`d`) + `cri` * MAD(`d`). 
* If `typcut` = `:q`, then `cutoff` is estimated from the empirical cumulative density function computed on `d`, for 
    a given risk-I (`alpha`).
Approximate parametric cutoffs have been proposed in the literature (e.g., Nomikos & MacGregor 1995, Hubert et al. 2005,
Pomerantsev 2008). Whatever the approximation method used, it is recommended to tune the cutoff depending on the 
detection objectives. 

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
yclatrain = yclatrain
Xtest = Xp[s, :]
Ytest = Y[s, :]
yclatest = yclatest 

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
model = occod(; cri = 2.5)
#model = occod(typcut = :mad, cri = 4)
#model = occod(typcut = :q, alpha = .01)
fit!(model, fitm0, Xtrain_in) 
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
    title = "OD", xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
""" 
occod(; kwargs...) = JchemoModel(occod, nothing, kwargs)

function occod(fitm, X; kwargs...)
    par = recovkw(ParOcc, kwargs).par 
    @assert in(par.typcut, [:mad, :q]) "Argument 'typcut' must be :mad or :q."
    @assert 0 <= par.alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."
    d = outod(fitm, X).d
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
        )
    Occod(d, fitm, e_cdf, cutoff, par)
end

"""
    predict(object::Occod, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occod, X)
    m = nro(X)
    ## Orthogonal distance
    E = xresid(object.fitm, X)
    d = rownorm(E)
    ## End
    d = DataFrame(
        d = d, 
        dstand = d / object.cutoff, 
        pval = pval(object.e_cdf, d)
        )
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i in eachindex(d.d)]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end


