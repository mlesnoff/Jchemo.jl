"""
    occstah(; kwargs...)
    occstah(X; kwargs...)
One-class classification (OCC) using the Stahel-Donoho outlierness.
* `fitm` : The preliminary model (e.g., object `fitm` returned by function `pcasvd`) that was fitted on 
    the training data assumed to represent the reference (= target) class.
* `X` : Training X-data (n, p) on which was fitted model `fitm`.
Keyword arguments:
* `nlv` : Nb. random directions on which `X` is projected. 
* `typcut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.
* `scal` : Boolean. If `true`, each column of `X` is scaled such as in function `outstah`.
* `seed` : Eventual seed for the `Random.MersenneTwister` generator (used when simulating
    random projcetion directions). 

OCC using outlierness `d` as defined in function `outstah`.

The directions used for projections are simulated by random binary (0/1) values. 

See function `occsd` for details on the cutoffs and outputs.

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

#### Fit the Occ model
model = occstah(; nlv = 5000, cri = 2, scal = true)
#model = occstah(; nlv = 5000, cri = 2, scal = true, seed = 1234)
fit!(model, Xtrain_in)
@names model 
fitm = model.fitm ;
@names fitm 
@head dtrain_in = fitm.d
cutoff = fitm.cutoff

d = dtrain_in.dstand
f, ax = plotxy(1:length(d), d; color = (:red, .3), size = (500, 300), xlabel = "Observation index",
    title = "Stahel-Donoho", ylabel = "Standardized distance")
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
group = vcat(fill("Train_in", ntrain_in), fill("Test_in", ntest_in), fill("Test_out", ntest_out))
color = [:purple, (:green, .7), (:red, .3)]
f, ax = plotxy(1:length(d), d, group; color, size = (500, 300), leg_title = "Type of obs.", 
    title = "Stahel-Donoho", xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
""" 
occstah(; kwargs...) = JchemoModel(occstah, nothing, kwargs)

function occstah(X; kwargs...) 
    par = recovkw(ParOccstah{Q}, kwargs).par 
    @assert in(par.typcut, [:mad, :q]) "Argument 'typcut' must be :mad or :q."
    @assert 0 <= par.alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."
    p = nco(X)
    V = rand(MersenneTwister(par.seed), 0:1, p, par.nlv)
    res = outstah(X, V; scal = par.scal)
    d = res.d
    if par.typcut == :mad
        cutoff = median(d) + par.cri * madv(d)
    elseif par.typcut == :q
        cutoff = quantile(d, 1 - par.alpha)
    end
    e_cdf = StatsBase.ecdf(d)
    d = DataFrame(
        d = d, 
        dstand = d / cutoff, 
        pval = pval(e_cdf, d)
        )
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
    fcscale!(T, res.mu, res.sigma)
    T .= abs.(T)
    d = similar(T, m)
    @inbounds for i in eachindex(d)
        d[i] = maximum(vrow(T, i))
    end
    d = DataFrame(
        d = d, 
        dstand = d / object.cutoff, 
        pval = pval(object.e_cdf, d)
        )
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i in eachindex(d.d)]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

