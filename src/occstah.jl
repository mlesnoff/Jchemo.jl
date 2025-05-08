"""
    occstah(; kwargs...)
    occstah(X; kwargs...)
One-class classification using the Stahel-Donoho outlierness.
* `X` : Training X-data (n, p).
Keyword arguments:
* `nlv` : Nb. random directions on which `X` is projected. 
* `cut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `cut` = `:mad`, a constant. See thereafter.
* `risk` : When `cut` = `:q`, a risk-I level. See thereafter.
* `scal` : Boolean. If `true`, each column of `X` is scaled such as in function `outstah`.

In this method, the outlierness `d` of a given observation is the Stahel-Donoho outlierness 
(see function `outstah`).

See function `occsd` for details on the outputs.

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
## - cla_train is the reference class (= 'in')
## - cla_test contains the observations to be predicted (i.e. to be 'in' or 'out' of cla_train) 
## Below, cla_train = "EEH", and two situations are considered as examples for cla_test:
cla_train = "EHH"
s = Ytrain.typ .== cla_train
Xtrain_fin = Xtrain[s, :]    
ntrain = nro(Xtrain_fin)
## Two situations
cla_test = "PEE"    # here test obs. should be classified 'out'
#cla_test = "EHH"   # here test obs. should be classified 'in'
s = Ytest.typ .== cla_test
Xtest_fin = Xtest[s, :] 
ntest = nro(Xtest_fin)
## Only used to compute error rates
ytrain_fin = repeat(["in"], ntrain)
if cla_test == cla_train
    ytest_fin = repeat(["in"], ntest)
else
    ytest_fin = repeat(["out"], ntest)
end
y_fin = vcat(ytrain_fin, ytest_fin)
## 
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

## Data description
nlv = 10
model = pcasvd(; nlv) 
fit!(model, Xtrain_fin) 
Ttrain = model.fitm.T
Ttest = transf(model, Xtest_fin)
T = vcat(Ttrain, Ttest)
i = 1
group = vcat(repeat(["Train"], ntrain), repeat(["Test"], ntest))
plotxy(T[:, i], T[:, i + 1], group; leg_title = "Type of obs.", xlabel = string("PC", i), 
    ylabel = string("PC", i + 1), title = string(cla_train, "-", cla_test)).f

#### Occ
## Training
model = occstah(; nlv = 5000, scal = true)
fit!(model, Xtrain_fin) 
@names model 
@names model.fitm 
@head model.fitm.V  # random projection directions 
@head res = model.fitm.d
d = res.dstand
f, ax = plotxy(1:length(d), d; size = (500, 300), xlabel = "Obs. index", 
    ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
## Prediction of Test
res = predict(model, Xtest_fin) ;
@names res
@head res.d
@head res.pred
tab(res.pred)
errp(res.pred, ytest_fin)
conf(res.pred, ytest_fin).cnt
d1 = model.fitm.d.dstand
d2 = res.d.dstand
d = vcat(d1, d2)
f, ax = plotxy(1:length(d), d, group; size = (500, 300), leg_title = "Class", 
    xlabel = "Obs. index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
""" 
occstah(; kwargs...) = JchemoModel(occstah, nothing, kwargs)

function occstah(X; kwargs...) 
    par = recovkw(ParOccstah, kwargs).par 
    @assert in(par.cut, [:mad, :q]) "Argument 'cut' must be :mad or :q."
    @assert 0 <= par.risk <= 1 "Argument 'risk' must ∈ [0, 1]."
    p = nco(X)
    V = rand(0:1, p, par.nlv)
    res = outstah(X, V; scal = par.scal)
    d = res.d
    ## Old parametric, not used anymore
    #d2 = d.^2 
    #mu = median(d2)
    #s2 = madv(d2)^2
    #nu = 2 * mu^2 / s2
    #g = mu / nu
    #dis = Distributions.Chisq(nu)
    #pval = Distributions.ccdf.(dis, d2 / g)
    #cut == :par ? cutoff = sqrt(g * quantile(dis, 1 - risk)) : nothing
    #cut == "npar" ? cutoff = median(d) + par.cri * madv(d) : nothing
    ## End 
    par.cut == :mad ? cutoff = median(d) + par.cri * madv(d) : nothing
    par.cut == :q ? cutoff = quantile(d, 1 - par.risk) : nothing
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val)
    Occstah(d, res, V, e_cdf, cutoff, par)
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
    T = zX * object.V
    fcenter!(T, res.mu)
    fscale!(T, res.s)
    T .= abs.(T)
    d = similar(T, m)
    @inbounds for i in axes(zX, 1)
        d[i] = maximum(vrow(T, i))
    end
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = p_val)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

