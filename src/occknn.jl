"""
    occknn(; kwargs...)
    occknn(X; kwargs...)
One-class classification using kNN distance-based outlierness.
* `X` : Training X-data (n, p) assumed to represent the reference class.
Keyword arguments:
* `nsamp` : Nb. of observations (`X`-rows) sampled in the training data and for which are computed 
    the outliernesses (stimated outlierness distribution of the reference class).
* `metric` : Metric used to compute the distances. See function `getknn`.
* `k` : Nb. nearest neighbors to consider.
* `algo` : Function summarizing the `k` distances to the neighbors.
* `cut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `cut` = `:mad`, a constant. See thereafter.
* `risk` : When `cut` = `:q`, a risk-I level. See thereafter.
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

See functions:
* `outknn` for details on the outlierness computation,
* and `occsd` for details on the the cutoff computation and the outputs.

When **predictions** are done, the outlierness of each new observation is compared to the outlierness 
distribution estimated from the `nsamp` sampled observations. 

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

## Data description
nlv = 10
model = pcasvd(; nlv) 
fit!(model, Xtrain_fin) 
Ttrain = model.fitm.T
Ttest = transf(model, Xtest_fin)
T = vcat(Ttrain, Ttest)
i = 1
group = vcat(repeat(["Train-EHH"], ntrain), repeat(["Test-EHH"], ntest1), repeat(["Test-PEE"], ntest2))
color = [:red, :blue, (:green, .5)]
plotxy(T[:, i], T[:, i + 1], group; color = color, leg_title = "Type of obs.", xlabel = string("PC", i), 
    ylabel = string("PC", i + 1)).f

#### Occ
## Training
nsamp = 150 ; k = 5 ; cri = 2.5
model = occknn(; nsamp, k, cri)
fit!(model, Xtrain_fin) 
@names model 
fitm = model.fitm ;
@names fitm 
@head dtrain = fitm.d
fitm.cutoff
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
group = vcat(repeat(["Train"], nsamp), repeat(["Test-EHH"], ntest1), repeat(["Test-PEE"], ntest2))
color = [:red, :blue, (:green, .5)]
f, ax = plotxy(1:length(d), d, group; color = color, size = (500, 300), leg_title = "Type of obs.", 
    xlabel = "Obs. index", ylabel = "Standardized distance")
hlines!(ax, 1; linestyle = :dot)
f
```
""" 
occknn(; kwargs...) = JchemoModel(occknn, nothing, kwargs)

function occknn(X; kwargs...)
    par = recovkw(ParOccknn, kwargs).par
    X = ensure_mat(X)
    Q = eltype(X)
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
        s = sample(1:n, nsamp, replace = false)
    end
    vX = vrow(X, s)
    par.k > n - 1 ? k = n - 1 : k = par.k
    ## kNN distance
    res = getknn(X, vX; k = k + 1, metric = par.metric)
    d = similar(X, nsamp)
    @inbounds for i in eachindex(d)
        d[i] = par.algo(res.d[i][2:end])
    end
    ## End 
    par.cut == :mad ? cutoff = median(d) + par.cri * madv(d) : nothing
    par.cut == :q ? cutoff = quantile(d, 1 - par.risk) : nothing
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val)
    Occknn(d, X, e_cdf, cutoff, xscales, par)
end

function predict(object::Occknn, X)
    X = ensure_mat(X)
    m = nro(X)
    ## kNN distance
    res = getknn(object.X, fscale(X, object.xscales); k = object.par.k + 1, 
        metric = object.par.metric) 
    d = similar(X, m)
    @inbounds for i in eachindex(d)
        d[i] = object.par.algo(res.d[i][2:end])
    end
    ## End
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = p_val)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

