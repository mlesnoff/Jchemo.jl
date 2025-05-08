"""
    occod(; kwargs...)
    occod(fitm, X; kwargs...)
One-class classification using PCA/PLS orthognal distance (OD).
* `fitm` : The preliminary model (e.g. object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the training class.
* `X` : Training X-data (n, p), on which was fitted the model `fitm`.
Keyword arguments:
* `cut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `cut` = `:mad`, a constant. See thereafter.
* `risk` : When `cut` = `:q`, a risk-I level. See thereafter.

In this method, the outlierness `d` of an observation is the orthogonal distance (=  'X-residuals') of this 
observation, ie. the Euclidean distance between the observation and its projection on the score plan defined by 
the fitted (e.g. PCA) model (e.g. Hubert et al. 2005, Van Branden & Hubert 2005 p. 66, Varmuza & Filzmoser 
2009 p. 79).

See function `occsd` for details on the outputs.

## References
M. Hubert, V. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust principal components 
analysis. Technometrics, 47, 64-79.

K. Vanden Branden, M. Hubert (2005). Robuts classification in high dimension based on the SIMCA method. 
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
Xtest = Xp[s, :]
Ytest = Y[s, :]

## Build the example data
## - cla_train is the reference class ('in')
## - cla_test contains the observations to be predicted (i.e. to be 'in' or 'out' of cla_train) 
## Below, cla_train = "EEH", and two situations are considered as example for cla_test:
cla_train = "EHH" ; cla_test = "PEE" ; typtest = "out"   # here cla_test should be classified 'out'
#cla_train = "EHH" ; cla_test = "EHH" ; typtest = "in"   # here cla_test should be classified 'in'
s = Ytrain.typ .== cla_train
zXtrain = Xtrain[s, :]    
s = Ytest.typ .== cla_test
zXtest = Xtest[s, :] 
ntrain = nro(zXtrain)
ntest = nro(zXtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)
ytrain = repeat(["in"], ntrain)
ytest = repeat([typtest], ntest)
## End

#### Preliminary PCA fitted model
nlv = 20
model_lv = pcasvd(; nlv) 
#model_lv = pcaout(; nlv) 
fit!(model_lv, zXtrain) 
res = summary(model_lv, zXtrain).explvarx 
plotgrid(res.nlv, res.pvar; step = 2, xlabel = "Nb. LVs", ylabel = "% Variance explained").f
Ttrain = model_lv.fitm.T
Ttest = transf(model_lv, zXtest)
T = vcat(Ttrain, Ttest)
i = 1
group = vcat(repeat(["Train"], ntrain), repeat(["Test"], ntest))
plotxy(T[:, i], T[:, i + 1], group; leg_title = "Class", xlabel = string("LV", i), 
    ylabel = string("LV", i + 1)).f

#### Occ
model = occod()
#model = occod(cut = :mad, cri = 4)
#model = occod(cut = :q, risk = .01)
#model = occsdod()
fit!(model, model_lv.fitm, zXtrain) 
@names model 
@names model.fitm 
@head dtrain = model.fitm.d
d = dtrain.dstand
f, ax = plotxy(1:length(d), d; size = (500, 300), xlabel = "Obs. index", 
    ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
## Predictions
res = predict(model, zXtest) 
@names res
@head pred = res.pred
@head dtest = res.d
tab(pred)
errp(pred, ytest)
conf(pred, ytest).cnt
##
d = vcat(dtrain.dstand, dtest.dstand)
f, ax = plotxy(1:length(d), d, group; size = (500, 300), leg_title = "Class", xlabel = "Obs. index", 
    ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
""" 
occod(; kwargs...) = JchemoModel(occod, nothing, kwargs)

function occod(fitm, X; kwargs...)
    par = recovkw(ParOcc, kwargs).par 
    @assert in(par.cut, [:mad, :q]) "Argument 'cut' must be :mad or :q."
    @assert 0 <= par.risk <= 1 "Argument 'risk' must ∈ [0, 1]."
    E = xresid(fitm, X)
    d = rownorm(E)
    if par.cut == :mad
        cutoff = median(d) + par.cri * madv(d)
    elseif par.cut == :q
        cutoff = quantile(d, 1 - par.risk)
    end
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val)
    Occod(d, fitm, e_cdf, cutoff, par)
end

"""
    predict(object::Occod, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occod, X)
    E = xresid(object.fitm, X)
    m = nro(E)
    d = rownorm(E)
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = p_val)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end


