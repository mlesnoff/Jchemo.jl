"""
    occknn(; kwargs...)
    occknn(X; kwargs...)
One-class classification (OCC) using kNN distance-based outlierness.
* `X` : Training X-data (n, p) assumed to represent the reference (= target) class.
Keyword arguments:
* `nsamp` : Nb. of observations (`X`-rows) sampled in the training data and for which are computed 
    the outliernesses (Monte Carlo simulation of the outlierness distribution of the reference class).
* `metric` : Metric used to compute the distances. See function `getknn`.
* `k` : Nb. nearest neighbors to consider.
* `algo` : Function summarizing the `k` distances to the neighbors.
* `typcut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.
* `scal` : Symbol defining the column scaling of `X`. Possible values are: `:none`, `std` (uncorrected STD) and `prt` (pareto).

OCC using outlierness `d` as defined in function `outknn`.

See function `occsd` for details on the cutoffs and outputs.

For predictions (`predict`), the outlierness of each new observation is compared to the outlierness 
distribution estimated from the `nsamp` observations sampled in the target class. 

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
nsamp = 150 ; k = 5 ; cri = 2.5
#nsamp = copy(ntrain_in) ; k = 5 ; cri = 2.5
model = occknn(; nsamp, k, cri)
#model = occknn(; nsamp, k, cri, seed = 1234)
#model = occlknn(; nsamp, k = 10, cri)
fit!(model, Xtrain_in)
@names model 
fitm = model.fitm ;
@names fitm 
@head dtrain_in = fitm.d   #  results for the 'nsamp' sampled training observations
cutoff = fitm.cutoff

d = dtrain_in.dstand
f, ax = plotxy(1:length(d), d; color = (:red, .3), size = (500, 300), xlabel = "Observation index",
    ylabel = "Standardized distance")
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
group = vcat(fill("Train_in", nsamp), fill("Test_in", ntest_in), fill("Test_out", ntest_out))
color = [:purple, (:green, .7), (:red, .3)]
f, ax = plotxy(1:length(d), d, group; color, size = (500, 300), leg_title = "Type of obs.", 
    xlabel = "Observation index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
""" 
occknn(; kwargs...) = JchemoModel(occknn, nothing, kwargs)

function occknn(X; kwargs...)
    par = recovkw(ParOccknn{Q}, kwargs).par
    X = ensure_mat(X)
    n, p = size(X)
    xscales = ones(Q, p)
    if par.scal
        xscales .= colstd(X)
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

