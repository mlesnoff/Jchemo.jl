"""
    outsd(fitm)
Compute an outlierness from PCA/PLS score distance (SD).
* `fitm` : The preliminary model (e.g. object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the training class.

In this method, the outlierness `d` of an observation is defined by its score distance (SD), ie. the Mahalanobis 
distance between the projection of the observation on the score plan defined by the fitted (e.g. PCA) model and the 
"center" (always defined by zero) of the score plan.

## References
M. Hubert, V. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust principal components 
analysis. Technometrics, 47, 64-79.

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
## - cla_test contains the observations to be predicted (to be 'in' or 'out' of cla_train) 
## Below, cla_train = "EEH", and two example situations for cla_test:
cla_train = "EHH" ; cla_test = "PEE" ; typtest = "out"   # here cla_test should be detected 'out'
#cla_train = "EHH" ; cla_test = "EHH" ; typtest = "in"   # here cla_test should not be detected 'out'
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
model = outsd()
#model = outsd(cut = :mad, cri = 4)
#model = outsd(cut = :q, risk = .01)
fit!(model, model_lv.fitm) 
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
function outsd(fitm)
    Q = eltype(fitm.T)
    nlv = nco(fitm.T)
    tscales = colstd(fitm.T, fitm.weights)
    T = fscale(fitm.T, tscales)
    d2 = vec(euclsq(T, zeros(Q, nlv)'))   # the center is defined as 0
    d = sqrt.(d2)
    (d = d,)
end
