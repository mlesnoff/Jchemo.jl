struct Occknndis
    d::DataFrame
    fm
    T::Array{Float64}
    scal::Vector{Float64}
    k::Int
    e_cdf
    cutoff::Real    
end

"""
    occknndis(X; 
        nsamp, nlv, k, 
        typc = "mad", cri = 3, alpha = .05)
One-class classification using "global" k-nearest neighbors distances.

* `X` : X-data.
* `nlv` : Nb. components for PCA.
* `nsamp` : Nb. of training observations (rows of `X`) used to compute the 
    empirical distribution of outlierness.
* `k` : Nb. of neighbors used to compute the outlierness.
* `typc` : Type of cutoff ("mad" or "q"). See below.
* `cri` : When `typc = "mad"`, constant used for computing the 
    cutoff detecting extreme values.
* `alpha` : When `typc = "q"`, risk-I level used for computing the cutoff 
    detecting extreme values.

The outlierness `d` of a given observation (training or test) is defined 
as the median of the distances of this observation to its `k` nearest neighbors
in the training. These distances are computed as Mahalanobis distances in a 
PCA score space.

A heuristic cutoff for detecting "extreme" `d` is computed 
from the empirical distribution of `d`. This distribution is computed from 
a number of `nsamp` observations randomly (and globally) sampled in the training.
* If `typc = "mad"`: The cutoff is computed by median(`d`) + `cri` * mad(`d`). 
* If `typc = "q"`: The cutoff is estimated from the empirical cdf of `d`. 

Column `dstand` is a standardized distance defined as `d / cutoff`. 
A value `dstand` > 1 may be considered as extreme compared to the distribution
of the training data. Column `pval` reports the p-values based on the 
empirical cdf. 

In the output `pred` of fonction `predict`, an observation is classified 
as 0 (i.e. belonging to the training class) when `dstand` <= 1 and 
1 (extreme) when `dstand` > 1. 

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "challenge2018.jld2") 
@load db dat
X = dat.X    
Y = dat.Y
f = 21 ; pol = 3 ; d = 2 ;
Xp = savgol(snv(X); f = f, pol = pol, d = d) 
s = Bool.(Y.test)
Xtrain = rmrow(Xp, s)
Ytrain = rmrow(Y, s)
Xtest = Xp[s, :]
Ytest = Y[s, :]

g1 = "EHH" ; g2 = "PEE"
#g1 = "EHH" ; g2 = g1
s1 = Ytrain.typ .== g1
s2 = Ytest.typ .== g2
zXtrain = Xtrain[s1, :]  
zXtest = Xtest[s2, :] 
ntrain = nro(zXtrain)
ntest = nro(zXtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

fm = pcasvd(zXtrain, nlv = 5) ; 
Ttrain = fm.T
Ttest = transform(fm, zXtest)
T = vcat(Ttrain, Ttest)
group = vcat(repeat(["0-Train"], ntrain), repeat(["1-Test"], ntest))
i = 1
plotxy(T[:, i], T[:, i + 1], group;
    xlabel = string("PC", i), ylabel = string("PC", i + 1)).f

#### End data

pct = .7
nlv = 30 ; k = Int64(round(pct * ntrain))
nsamp = 300
fm = occknndis(zXtrain; nsamp = nsamp,
    nlv = nlv, k = k, typc = "mad") ;
fm.d
hist(fm.d.dstand; bins = 50)

res = Jchemo.predict(fm, zXtest) ;
res.d
res.pred
tab(res.pred)

d1 = fm.d.dstand
d2 = res.d.dstand
d = vcat(d1, d2)
f, ax = plotxy(1:length(d), d)
hlines!(ax, 1)
f
```
""" 
function occknndis(X; 
        nsamp, nlv, k, 
        typc = "mad", cri = 3, alpha = .05)
    X = ensure_mat(X)
    n = nro(X)
    fm = pcasvd(X; nlv = nlv)
    T = fm.T
    scal = colstd(T)
    scale!(T, scal)
    zn = collect(1:n)
    samp = sample(zn, nsamp; replace = false)
    res = getknn(T, T[samp, :]; 
            k = k + 1, metric = "eucl")
    d = zeros(nsamp)
    @inbounds for i = 1:nsamp
        d[i] = median(res.d[i][2:end])
    end
    typc == "mad" ? cutoff = median(d) + cri * mad(d) : nothing
    typc == "q" ? cutoff = quantile(d, 1 - alpha) : nothing
    e_cdf = StatsBase.ecdf(d)
    pval = 1 .- e_cdf(d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = pval)
    Occknndis(d, fm, T, scal, k, e_cdf, cutoff)
end

function predict(object::Occknndis, X)
    X = ensure_mat(X)
    m = size(X, 1)
    T = transform(object.fm, X)
    scale!(T, object.scal)
    res = getknn(object.T, T; k = object.k, metric = "eucl") 
    d = zeros(m)
    @inbounds for i = 1:m
        d[i] = median(res.d[i])
    end
    pval = 1 .- object.e_cdf(d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = pval)
    pred = reshape(Int64.(d.dstand .> 1), m, 1)
    (pred = pred, d)
end




