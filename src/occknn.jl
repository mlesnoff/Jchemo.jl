"""
    occknn(; kwargs...)
    occknn(X; kwargs...)
One-class classification (OCC) using kNN distance-based outlierness.
* `X` : Training X-data (n, p) assumed to represent the reference (= target) class.
Keyword arguments:
* `nsamp` : Nb. of observations (`X`-rows) sampled in the training data used to estimate 
    the reference outlierness distribution (i.e., the disitribution of the outlierness of observations belonging 
    to the reference class). The sampling is random with no replacement. If `nsamp` = n, all the training 
    observations are used to estimate this distribution.
* `metric` : Metric used to compute the distances. See function `getknn`.
* `k` : Nb. nearest neighbors to consider.
* `algo` : Function summarizing the distances to the `k` neighbors.
* `typcut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.
* `scal` : Symbol defining the column scaling of `X`. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

The general principle is in two steps: 
1) The distribution of the outlierness of the reference class is estimated by Monte Carlo: `nsamp` 
    observations are sampled in `X` and their outlierness is computed; 
2) For each new observation to predict (function `predict`), its outlierness is computed and compared to the 
    reference distribution. If this outlierness is larger than a cutoff computed from the reference distribution
    (e.g., defined by a quantile), the observation is predicted as 'out' (i.e., not belonging to the reference 
    class), or 'in' otherwise. 

The method to compute outlierness is defined in function `outknn` (see for details). 
See also function `occsd` for the possible cutoff types and the outputs.

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
nsamp = 150 ; k = 5 ; cri = 2.5
#nsamp = copy(nref) ; k = 5 ; cri = 2.5
model = occknn(; nsamp, k, cri)
#model = occknn(; nsamp, k, cri, seed = 1234)
#model = occlknn(; nsamp, k = 10, cri)
fit!(model, Xref)
@names model 
fitm = model.fitm ;
@names fitm 
@head dref = fitm.d   #  results for the 'nsamp' sampled training observations
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
group = vcat(fill("1-Train (ref)", nsamp), fill("2-New_ref", nnew_ref), fill("3-New_out", nnew_out))
color = [(:red, .3), (:green, .5), :purple]
f, ax = plotxy(1:length(d), d, group; color = color, size = (500, 300), leg_title = "Type of obs.", 
    xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
""" 
occknn(; kwargs...) = JchemoModel(occknn, nothing, kwargs)

function occknn(X; kwargs...)
    X = ensure_mat(X)
    n, p = size(X)    
    Q = eltype(X)
    par = recovkw(ParOccknn{Q}, kwargs).par
    xscales = ones(Q, p)
    if par.scal != :none
        colscal = def_colscal(par.scal) 
        xscales .= colscal(X, weights)
        X = fscale(X, xscales)
    end
    nsamp = min(par.nsamp, n)
    if nsamp == n
        s = 1:n
    else
        s = sample(MersenneTwister(par.seed), 1:n, nsamp; replace = false)
    end
    vX = vrow(X, s)
    k = min(par.k, n - 1)
    ## Distribution of outlierness of the 'nsamp' sampled training observations
    res = getknn(X, vX; k = k + 1, metric = par.metric)
    d = similar(X, nsamp)
    @inbounds for i in eachindex(d)
        d[i] = par.algo(res.d[i][2:end])
    end
    ## End 
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
    Occknn(d, X, e_cdf, cutoff, xscales, par)
end

function predict(object::Occknn, X)
    X = ensure_mat(X)
    m = nro(X)
    ## kNN distance
    res = getknn(object.X, fscale(X, object.xscales); k = object.par.k + 1, metric = object.par.metric) 
    d = similar(X, m)
    @inbounds for i in eachindex(d)
        d[i] = object.par.algo(res.d[i][2:end])
    end
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

