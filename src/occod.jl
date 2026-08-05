"""
    occod(; kwargs...)
    occod(fitm, X; kwargs...)
One-class classification (OCC) using PCA/PLS orthognal distance (OD).
* `fitm` : The preliminary model (e.g., object `fitm` returned by functions `pcasvd` or `plskern`) 
    that was fitted on the training data assumed to represent the reference (= target) class.
* `X` : Training X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to consider. By default, it is the maximum nb. of LVs
    defined in model `object`.
* `typcut` : Type of cutoff. Possible values are: `:std`, `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:std` or `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.

OCC using outlierness `d` as defined in function `outod`.

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
color = [(:red, .3), (:green, .5), :purple]
i = 1
plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], group; color = color, leg_title = "Type of obs.", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1), zlabel = string("PC", i + 2)).f

#### Fit the Occ model based on the fitted score space 
model = occod(cri = 2.5)
#model = occod(typcut = :std, cri = 2.5)
#model = occod(typcut = :q, alpha = .01)
#model = occod(nlv = 5, cri = 2.5)
fit!(model, fitm0, Xref)
@names model 
fitm = model.fitm ;
@names fitm 
@head dref = fitm.d
fitm.cutoff

d = dref.dstand
s = d .> 1
f, ax = plotxy(1:length(d), d;color = (:red, .3), size = (500, 300), title = "Train (reference class)",  
    xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
scatter!(ax, (1:length(d))[s], d[s]; color = :red, label = "Extreme")
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
color = [(:red, .3), (:green, .5), :purple]
f, ax = plotxy(1:length(d), d, group; color = color, size = (500, 300), leg_title = "Type of obs.", 
    title = "OD", xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
""" 
occod(; kwargs...) = JchemoModel(occod, nothing, kwargs)

function occod(fitm, X; kwargs...)
    X = ensure_mat(X)
    Q = eltype(X)
    par = recovkw(ParOcc{Q}, kwargs).par 
    @assert in(par.typcut, [:std, :mad, :q]) "Argument 'typcut' must be :std, :mad or :q."
    @assert 0 <= par.alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."
    if isnothing(par.nlv)
        par.nlv = nco(fitm.T)
    else
        par.nlv = min(par.nlv, nco(fitm.T))
    end
    d = outod(fitm, X; par.nlv).d
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
    E = xresid(object.fitm, X, object.par.nlv)
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


