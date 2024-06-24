"""
    occod(fm, X; kwargs...)
One-class classification using PCA/PLS orthognal distance (OD).
* `fm` : The preliminary model that (e.g. PCA) was fitted 
    (object `fm`) on the training data assumed to represent 
    the training class.
* `X` : Training X-data (n, p), on which was fitted 
    the model `fm`.
Keyword arguments:
* `mcut` : Type of cutoff. Possible values are: `:mad`, `:q`. 
    See Thereafter.
* `cri` : When `mcut` = `:mad`, a constant. See thereafter.
* `risk` : When `mcut` = `:q`, a risk-I level. See thereafter.

In this method, the outlierness `d` of an observation
is the orthogonal distance (OD =  "X-residuals") of this 
observation, ie. the Euclidean distance between the observation 
and its projection on the  score plan defined by the fitted 
(e.g. PCA) model (e.g. Hubert et al. 2005, Van Branden & Hubert 
2005 p. 66, Varmuza & Filzmoser 2009 p. 79).

See function `occsd` for details on outputs.

## References
M. Hubert, P. J. Rousseeuw, K. Vanden Branden (2005). 
ROBPCA: a new approach to robust principal components 
analysis. Technometrics, 47, 64-79.

K. Vanden Branden, M. Hubert (2005). Robuts classification 
in high dimension based on the SIMCA method. Chem. Lab. Int. 
Syst, 79, 10-21.

K. Varmuza, P. Filzmoser (2009). Introduction to multivariate 
statistical analysis in chemometrics. CRC Press, Boca Raton.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/challenge2018.jld2") 
@load db dat
pnames(dat)
X = dat.X    
Y = dat.Y
mod = model(savgol; npoint = 21, deriv = 2, degree = 3)
fit!(mod, X) 
Xp = transf(mod, X) 
s = Bool.(Y.test)
Xtrain = rmrow(Xp, s)
Ytrain = rmrow(Y, s)
Xtest = Xp[s, :]
Ytest = Y[s, :]

## Below, the reference class is "EEH"
cla1 = "EHH" ; cla2 = "PEE" ; cod = "out"   # here cla2 should be detected
#cla1 = "EHH" ; cla2 = "EHH" ; cod = "in"   # here cla2 should not be detected
s1 = Ytrain.typ .== cla1
s2 = Ytest.typ .== cla2
zXtrain = Xtrain[s1, :]    
zXtest = Xtest[s2, :] 
ntrain = nro(zXtrain)
ntest = nro(zXtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)
ytrain = repeat(["in"], ntrain)
ytest = repeat([cod], ntest)

## Group description
mod0 = model(pcasvd; nlv = 10) 
fit!(mod, zXtrain) 
Ttrain = mod0.fm.T
Ttest = transf(mod0, zXtest)
T = vcat(Ttrain, Ttest)
group = vcat(repeat(["1"], ntrain), repeat(["2"], ntest))
i = 1
plotxy(T[:, i], T[:, i + 1], group; leg_title = "Class", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1)).f

#### Occ
## Preliminary PCA fitted model
mod0 = model(pcasvd; nlv = 10) 
fit!(mod0, zXtrain)
fm0 = mod0.fm ;  
## Outlierness
mod = model(occod)
#mod = model(occod; mcut = :mad, cri = 4)
#mod = model(occod; mcut = :q, risk = .01) ;
#mod = model(occsdod)
fit!(mod, fm0, zXtrain) 
pnames(mod) 
pnames(mod.fm) 
@head d = mod.fm.d
d = d.dstand
f, ax = plotxy(1:length(d), d; size = (500, 300), 
    xlabel = "Obs. index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f

res = predict(mod, zXtest) ;
pnames(res)
@head res.d
@head res.pred
tab(res.pred)
errp(res.pred, ytest)
conf(res.pred, ytest).cnt
d1 = mod.fm.d.dstand
d2 = res.d.dstand
d = vcat(d1, d2)
f, ax = plotxy(1:length(d), d, group; size = (500, 300), leg_title = "Class", 
    xlabel = "Obs. index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
""" 
function occod(fm, X; kwargs...)
    par = recovkwargs(Par, kwargs) 
    @assert 0 <= par.risk <= 1 "Argument 'risk' must ∈ [0, 1]."
    E = xresid(fm, X)
    d2 = vec(sum(E .* E, dims = 2))
    d = sqrt.(d2)
    par.mcut == :mad ? cutoff = median(d) + par.cri * mad(d) : nothing
    par.mcut == :q ? cutoff = quantile(d, 1 - par.risk) : nothing
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val)
    Occod(d, fm, e_cdf, cutoff)
end

"""
    predict(object::Occod, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occod, X)
    E = xresid(object.fm, X)
    m = nro(E)
    d2 = vec(sum(E .* E, dims = 2))
    d = sqrt.(d2)
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, 
        pval = p_val)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end


