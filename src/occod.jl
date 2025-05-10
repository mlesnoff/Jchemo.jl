"""
    occod(; kwargs...)
    occod(fitm, X; kwargs...)
One-class classification using PCA/PLS orthognal distance (OD).
* `fitm` : The preliminary model (e.g. object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the reference class.
* `X` : Training X-data (n, p), on which was fitted the model `fitm`.
Keyword arguments:
* `cut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `cut` = `:mad`, a constant. See thereafter.
* `risk` : When `cut` = `:q`, a risk-I level. See thereafter.

In this method, the outlierness `d` of an observation is the orthogonal distance (=  'X-residuals') of this 
observation, ie. the Euclidean distance between the observation and its projection to the score plan defined by 
the fitted (e.g. PCA) model (e.g. Hubert et al. 2005, Van Branden & Hubert 2005 p. 66, Varmuza & Filzmoser 
2009 p. 79).

See function `occsd` for details on the cutoff computation and the outputs.

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
## - cla_train is the reference class (= 'in'), "EHH" 
cla_train = "EHH"
s = Ytrain.typ .== cla_train
Xtrain_fin = Xtrain[s, :]    
ntrain = nro(Xtrain_fin)
## cla_test contains the observations to be predicted (i.e. to be 'in' or 'out' of cla_train), 
## a mix of "EEH" and "PEE" 
cla_test1 = "EHH"   # should be predicted 'in'
s = Ytest.typ .== cla_test1
Xtest_fin1 = Xtest[s, :] 
ntest1 = nro(Xtest_fin1)
##
cla_test2 = "PEE"   # should be predicted 'out'
s = Ytest.typ .== cla_test2
Xtest_fin2 = Xtest[s, :] 
ntest2 = nro(Xtest_fin2)
##
Xtest_fin = vcat(Xtest_fin1, Xtest_fin2)
## Only used to compute error rates
ytrain_fin = repeat(["in"], ntrain)
ytest_fin = [repeat(["in"], ntest1); repeat(["out"], ntest2)]
y_fin = vcat(ytrain_fin, ytest_fin)
## 
ntot = ntrain + ntest1 + ntest2
(ntot = ntot, ntrain, ntest1, ntest2)

#### Preliminary PCA fitted model
nlv = 15
model0 = pcasvd(; nlv) 
#model0 = pcaout(; nlv) 
fit!(model0, Xtrain_fin) 
res = summary(model0, Xtrain_fin).explvarx 
plotgrid(res.nlv, res.pvar; step = 2, xlabel = "Nb. LVs", ylabel = "% Variance explained").f
Ttrain = model0.fitm.T
Ttest = transf(model0, Xtest_fin)
T = vcat(Ttrain, Ttest)
i = 1
group = vcat(repeat(["Train-EHH"], ntrain), repeat(["Test-EHH"], ntest1), repeat(["Test-PEE"], ntest2))
color = [:red, :blue, (:green, .5)]
plotxy(T[:, i], T[:, i + 1], group; color = color, leg_title = "Type of obs.", xlabel = string("PC", i), 
    ylabel = string("PC", i + 1)).f

#### Occ
## Training
model = occod(; cri = 2.5)
#model = occod(cut = :mad, cri = 4)
#model = occod(cut = :q, risk = .01)
#model = occsdod(; cri = 2.5)
fit!(model, model0.fitm, Xtrain_fin) 
@names model 
fitm = model.fitm ;
@names fitm 
@head dtrain = fitm.d
#fitm.cutoff
d = dtrain.dstand
f, ax = plotxy(1:length(d), d; color = (:green, .5), size = (500, 300), xlabel = "Obs. index", 
    ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
## Prediction of Test
res = predict(model, Xtest_fin) 
@names res
@head pred = res.pred
@head dtest = res.d
tab(pred)
errp(pred, ytest_fin)
conf(pred, ytest_fin).cnt
##
d = vcat(dtrain.dstand, dtest.dstand)
color = [:red, :blue, (:green, .5)]
f, ax = plotxy(1:length(d), d, group; color = color, size = (500, 300), leg_title = "Type of obs.", 
    xlabel = "Obs. index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
""" 
occod(; kwargs...) = JchemoModel(occod, nothing, kwargs)

function occod(fitm, X; kwargs...)
    par = recovkw(ParOcc, kwargs).par 
    @assert in(par.cut, [:mad, :q]) "Argument 'cut' must be :mad or :q."
    @assert 0 <= par.risk <= 1 "Argument 'risk' must ∈ [0, 1]."
    ## Orthogonal distance
    E = xresid(fitm, X)
    d = rownorm(E)
    ## End
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
    m = nro(X)
    ## Orthogonal distance
    E = xresid(object.fitm, X)
    d = rownorm(E)
    ## End
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = p_val)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end


