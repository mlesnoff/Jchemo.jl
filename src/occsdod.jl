"""
    occsdod(; kwargs...)
    occsdod(object, X; kwargs...)
One-class classification (OCC) using a consensus between PCA/PLS score and orthogonal distances (SD and OD).
* `fitm` : The preliminary model (e.g., object `fitm` returned by functions `pcasvd` or `plskern`) 
    that was fitted on the training data assumed to represent the reference (= target) class.
* `X` : Training X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to consider. By default, it is the maximum nb. of LVs
    defined in model `object`.
* `typcut` : Type of cutoff. Possible values are: `:std`, `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:std` or `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.
* `gamma` : Proportion of scaled SD in the consensus (see function `outsdod`).
* `fscal` : Function used to scale SD and OD in the consensus (by default, this is `stdv`; see function `outsdod`). 

OCC using outlierness `d` as defined in function `outsdod`.

See function `occsd` for details on the cutoff types and the outputs.

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
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
i = 1
plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], group; color, leg_title = "Type of obs.", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1), zlabel = string("PC", i + 2)).f

#### Fit the Occ model based on the fitted score space 
model = occsdod(cri = 2.5)
#model = occsdod(typcut = :q, alpha = .01)
#model = occsdod(typcut = :std, cri = 2.5, fscal = stdv)
#model = occsdod(nlv = 5, cri = 2.5)
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
sdsigma = dref.sdsigma
odsigma = dref.odsigma
a = fitm.coefs[1]
b = fitm.coefs[2]
s = d .> cutoff
tsp = .4 ; color = (:orange, tsp)
f, ax = plotxy(sdsigma, odsigma; color, title = "Train (reference class)", xlabel = "SD / sigma", 
    ylabel = "OD / sigma")
scatter!(ax, sdsigma[s], odsigma[s]; color = color[1], label = "Extreme")
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

d = dref.d
a = fitm.coefs[1]
b = fitm.coefs[2]
s = d .> cutoff
tsp = .5 ; color = [(:orange, tsp), (:green, tsp), (:purple, tsp)]
f, ax = plotxy(sdsigma, odsigma; size = (600, 300), color = color[1], xlabel = "SD / sigma", 
    ylabel = "SD / sigma", label = "1-Train (ref)")
scatter!(ax, dnew_ref.sdsigma, dnew_ref.odsigma; color = color[2], label = "2-New_ref")
scatter!(ax, dnew_out.sdsigma, dnew_out.odsigma; color = color[3], label = "3-New_out")
ablines!(ax, a, b; color = :grey, linewidth = .7, linestyle = :dash, label = "Cutoff")
f[1, 2] = Legend(f, ax, "Type of obs."; framevisible = false)
f
```
""" 
occsdod(; kwargs...) = JchemoModel(occsdod, nothing, kwargs)

function occsdod(fitm, X; kwargs...) 
    X = ensure_mat(X) 
    Q = eltype(X)
    par = recovkw(ParOccsdod{Q}, kwargs).par 
    gamma = par.gamma
    @assert in(par.typcut, [:std, :mad, :q]) "Argument 'typcut' must be :std, :mad or :q."
    @assert 0 <= par.alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."
    if isnothing(par.nlv)
        par.nlv = nco(fitm.T)
    else
        par.nlv = min(par.nlv, nco(fitm.T))
    end
    sd = outsd(fitm; par.nlv)
    od = outod(fitm, X; par.nlv)
    sdod = outsdod(fitm, X; par.nlv, gamma, fscal = par.fscal)
    sigma_sd = sdod.sigma_sd 
    sigma_od = sdod.sigma_od
    ##
    d = sdod.d
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
        sd = sd.d,
        od = od.d,
        sdsigma = sd.d /  sigma_sd,
        odsigma = od.d / sigma_od,
        gh = sd.d.^2 / par.nlv
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
    X = ensure_mat(X)
    nlv = object.par.nlv
    gamma = object.par.gamma 
    tscales = object.sd.tscales    
    sigma_sd = object.sdod.sigma_sd
    sigma_od = object.sdod.sigma_od
    ## SD
    T = transf(object.fitm, X, nlv)
    Q = eltype(T)
    m = nro(T)
    fscale!(T, tscales)
    sd2 = vec(eucl2(T, zeros(Q, 1, nlv)))
    sd = sqrt.(sd2)
    ## OD
    E = xresid(object.fitm, X, nlv)
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

