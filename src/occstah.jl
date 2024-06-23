"""
    occstah(X; kwargs...)
One-class classification using the Stahel-Donoho outlierness.
* `X` : Training X-data (n, p).
Keyword arguments:
* `nlv` : Nb. dimensions on which `X` is projected. 
* `mcut` : Type of cutoff. Possible values are: `:mad`, `:q`. 
    See Thereafter.
* `cri` : When `mcut` = `:mad`, a constant. See thereafter.
* `risk` : When `mcut` = `:q`, a risk-I level. See thereafter.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled such as in function `outstah`.

In this method, the outlierness `d` of a given observation
is the Stahel-Donoho outlierness (see `?outstah`).

See function `occsd` for details on outputs.

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
mod = model(pcasvd; nlv = 10) 
fit!(mod, zXtrain) 
Ttrain = mod.fm.T
Ttest = transf(mod, zXtest)
T = vcat(Ttrain, Ttest)
group = vcat(repeat(["1"], ntrain), repeat(["2"], ntest))
i = 1
plotxy(T[:, i], T[:, i + 1], group; leg_title = "Class", 
    xlabel = string("PC", i), ylabel = string("PC", i + 1)).f

#### Occ
## Preliminary dimension 
## Not required but often more 
## efficient
nlv = 50
mod0 = model(pcasvd; nlv) ;
fit!(mod0, zXtrain)
Ttrain = mod0.fm.T
Ttest = transf(mod0, zXtest)
## Outlierness
mod = model(occstah; nlv, scal = true)
fit!(mod, Ttrain) 
pnames(mod) 
pnames(mod.fm) 
@head d = mod.fm.d
d = d.dstand
f, ax = plotxy(1:length(d), d; size = (500, 300), xlabel = "Obs. index", 
    ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f

res = predict(mod, Ttest) ;
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
function occstah(X; kwargs...) 
    par = recovkwargs(Par, kwargs) 
    @assert 0 <= par.risk <= 1 "Argument 'risk' must ∈ [0, 1]."
    p = nco(X)
    P = rand(0:1, p, par.nlv)
    res = outstah(X, P; scal = par.scal)
    d = res.d
    #d2 = d.^2 
    #mu = median(d2)
    #s2 = mad(d2)^2
    #nu = 2 * mu^2 / s2
    #g = mu / nu
    #dist = Distributions.Chisq(nu)
    #pval = Distributions.ccdf.(dist, d2 / g)
    #mcut == :par ? cutoff = sqrt(g * quantile(dist, 1 - risk)) : nothing
    #mcut == "npar" ? cutoff = median(d) + par.cri * mad(d) : nothing  
    par.mcut == :mad ? cutoff = median(d) + par.cri * mad(d) : nothing
    par.mcut == :q ? cutoff = quantile(d, 1 - par.risk) : nothing
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val)
    Occstah(d, res, P, e_cdf, cutoff)
end

"""
    predict(object::Occstah, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occstah, X)
    zX = copy(ensure_mat(X))  # for fscale!
    m = nro(zX)
    res = object.res_stah
    fscale!(zX, res.xscales)
    T = zX * object.P
    fcenter!(T, res.mu)
    fscale!(T, res.s)
    T .= abs.(T)
    d = similar(T, m)
    @inbounds for i = 1:m
        d[i] = maximum(vrow(T, i))
    end
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = p_val)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

