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

#### Fit the Occ model
model = occstah(; nlv = 5000, cri = 2., scal = :std)
#model = occstah(; nlv = 5000, cri = 2., scal = :std, seed = 1234)
fit!(model, Xref)
@names model 
fitm = model.fitm ;
@names fitm 
@head dref = fitm.d
cutoff = fitm.cutoff

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
group = vcat(fill("1-Train (ref)", nref), fill("2-New_ref", nnew_ref), fill("3-New_out", nnew_out))
color = [(:red, .3), (:green, .5), :purple]
f, ax = plotxy(1:length(d), d, group; color = color, size = (500, 300), leg_title = "Type of obs.", 
    title = "Stahel-Donoho", xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f

```
""" 
occstah(; kwargs...) = JchemoModel(occstah, nothing, kwargs)

function occstah(X; kwargs...)
    X = ensure_mat(X)
    p = nco(X)    
    Q = eltype(X)
    par = recovkw(ParOccstah{Q}, kwargs).par 
    @assert in(par.typcut, [:mad, :q]) "Argument 'typcut' must be :mad or :q."
    @assert 0 <= par.alpha <= 1 "Argument 'alpha' must ∈ [0, 1]."
    V = rand(MersenneTwister(par.seed), Q.(0:1), p, par.nlv)
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
    vX = copy(ensure_mat(X))  # for fscale!
    m = nro(vX)
    res = object.res_stah
    fscale!(vX, res.xscales)
    T = vX * object.V
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

