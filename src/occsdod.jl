"""
    occsdod(; kwargs...)
    occsdod(object, X; kwargs...)
One-class classification (OCC) using a consensus between PCA/PLS score and orthogonal distances (SD and OD).
* `fitm` : The preliminary model (e.g., object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the reference (= target) class.
* `X` : Training X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `typcut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.
* `gamma` : Proportion of scaled SD in the consensus (see function `outsdod`).
* `fscal` : Function used to scale SD and OD in the consensus (by default, this is `madv`; see function `outsdod`). 

OCC using outlierness `d` as defined in function `outsdod`.

See function `occsd`for details.  

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
ytrain_in = fill("in", ntrain_in)
ytest_in = fill("in", ntest_in)
ytest_out = fill("out", ntest_out)

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
group = vcat(fill("Train_in", ntrain_in), fill("Test_in", ntest_in), fill("Test_out", ntest_out))
color = [:purple, (:green, .7), (:red, .3)]
i = 1
plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], group; color, leg_title = "Type of obs.", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1), zlabel = string("PC", i + 2)).f

#### Fit the Occ model based on the fitted score space 'in' 
model = occsdod(cri = 2.5)
#model = occsdod(cri = 2.5, fscal = stdv)
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
a = fitm.coefs[1]
b = fitm.coefs[2]
s = d .> cutoff
f, ax = plotxy(sdsigma, odsigma; xlabel = "SD / sigma", ylabel = "OD / sigma")
scatter!(ax, sdsigma[s], odsigma[s]; color = :red, label = "Extreme")
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
f, ax = plotxy(1:length(d), d, group; color, size = (500, 300), leg_title = "Type of obs.", 
    title = "OD", xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f

d = dtrain_in.d
a = fitm.coefs[1]
b = fitm.coefs[2]
s = d .> cutoff
f, ax = plotxy(sdsigma, odsigma; xlabel = "SD / sigma", ylabel = "OD / sigma")
scatter!(ax, sdsigma[s], odsigma[s]; color = :red, label = "Train-In Extreme")
scatter!(ax, dtest_in.sdsigma, dtest_in.odsigma; color = (:purple, .5), label = "Test-In")
scatter!(ax, dtest_out.sdsigma, dtest_out.odsigma; color = :green, label = "Test-Out")
ablines!(ax, a, b; color = :red, linewidth = .7, linestyle = :dash)
axislegend(ax; position = :rb)
f
```
""" 
occsdod(; kwargs...) = JchemoModel(occsdod, nothing, kwargs)

function occsdod(fitm, X; kwargs...) 
    par = recovkw(ParOccsdod{Q}, kwargs).par 
    gamma = par.gamma
    @assert 0 <= gamma <= 1 "Argument 'gamma' must ∈ [0, 1]."   
    nlv = nco(fitm.T) 
    sd = outsd(fitm)
    od = outod(fitm, X)
    sdod = outsdod(fitm, X; gamma, fscal = par.fscal)
    sigma_sd = sdod.sigma_sd 
    sigma_od = sdod.sigma_od
    ##
    d = sdod.d
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
        sd = sd.d,
        od = od.d,
        sdsigma = sd.d /  sigma_sd,
        odsigma = od.d / sigma_od,
        gh = sd.d.^2 / nlv
        )
    ## Coefs for graphic SD/sigma - OD/sigma
    #a = cutoff * sigma_od / (1 - gamma)
    #b = -gamma / (1 - gamma) * sigma_od / sigma_sd
    a = cutoff / (1 - gamma)
    b = -gamma / (1 - gamma)
    coefs = [a; b]
    ##
    Occsdod(d, fitm, e_cdf, cutoff, sd, od, sdod, coefs, par)  
end

"""
    predict(object::Occsdod, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occsdod, X)
    tscales = object.sd.tscales    
    gamma = object.par.gamma 
    sigma_sd = object.sdod.sigma_sd
    sigma_od = object.sdod.sigma_od
    ## SD
    T = transf(object.fitm, X)
    Q = eltype(T)
    m, nlv = size(T)
    fscale!(T, tscales)
    sd2 = vec(eucl2(T, zeros(Q, nlv)'))
    sd = sqrt.(sd2)
    ## OD
    E = xresid(object.fitm, X)
    od = rownorm(E)
    ## Consensus
    d = gamma * sd / sigma_sd + (1 - gamma) * od / sigma_od
    ## End
    d = DataFrame(
        d = d, 
        dstand = d / object.cutoff, 
        pval = pval(object.e_cdf, d),
        sd = sd,
        od = od, 
        sdsigma = sd / object.sdod.sigma_sd,
        odsigma = od / object.sdod.sigma_od,
        gh = sd2 / nlv
        )
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i in eachindex(d.d)]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

