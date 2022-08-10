struct Occlknndis
    d::DataFrame
    fm
    T::Array{Float64}
    scal::Vector{Float64}
    k::Int
    e_cdf
    cutoff::Real    
end

"""
    occlknndis(X; 
        nsamp, nlv, k, 
        typc = "mad", cri = 3, alpha = .05)
One-class classification using "local" k-nearest neighbors distances.

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

In this method, the "outlierness measure" `d` of a given observatio "q"  
is defined as follows: 
* The median of the distances between "q" and its `k` nearest neighbors
in the training is computed, say d_q. 
* Then, this same median distance is computed for each of the `k` nearest
neighbors of "q" (i.e. each of the `k` nearest neighbors of "q" becomes 
successively an observation "q" from which is computed the median distance to its
`k` neighbors). This returns `k` median distances. Finally, the median of these
`k` median distances is computed, and say d_q_nn.
* The outlierness of "q" is defined by d_q / d_q_nn.  
The distances are computed as Mahalanobis distances in a PCA score space 
(internally computed; argument `nlv`).

See `?occknndis` for the cutoff computation (the same principle is applied). 

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

nlv = 30 ; k = 5
nsamp = 50
fm = Jchemo.occlknndis(zXtrain; nsamp = nsamp,
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
function occlknndis(X; 
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
    ind = Int64.(zeros(k))
    d = zeros(nsamp)
    zd = zeros(nsamp)
    vd = zeros(k)
    @inbounds for i = 1:nsamp
        # view vrow(T, i) does not work with getknn
        res = getknn(rmrow(T, samp[i]), T[i:i, :]; k = k, metric = "eucl")
        ind .= res.ind[1]
        d[i] = median(res.d[1])
        for j = 1:k
            zind = ind[j]
            zres = getknn(rmrow(T, zind), T[zind:zind, :]; k = k, metric = "eucl")
            vd[j] = median(zres.d[1])
        end
        zd[i] = median(vd)
    end
    d ./= zd
    typc == "mad" ? cutoff = median(d) + cri * mad(d) : nothing
    typc == "q" ? cutoff = quantile(d, 1 - alpha) : nothing
    e_cdf = StatsBase.ecdf(d)
    pval = 1 .- e_cdf(d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = pval)
    Occlknndis(d, fm, T, scal, k, e_cdf, cutoff)
end

function predict(object::Occlknndis, X)
    X = ensure_mat(X)
    m = size(X, 1)
    T = transform(object.fm, X)
    scale!(T, object.scal)
    res = getknn(object.T, T; k = object.k, metric = "eucl")
    ind = Int64.(zeros(object.k))
    d = zeros(m)
    zd = zeros(m)
    vd = zeros(object.k)
    @inbounds for i = 1:m
        ind .= res.ind[i]
        d[i] = median(res.d[i])
        for j = 1:object.k
            zind = ind[j]
            zres = getknn(rmrow(object.T, zind), object.T[zind:zind, :]; 
                k = object.k, metric = "eucl")
            vd[j] = median(zres.d[1])
        end 
        zd[i] = median(vd)
    end
    d ./= zd
    pval = 1 .- object.e_cdf(d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = pval)
    pred = reshape(Int64.(d.dstand .> 1), m, 1)
    (pred = pred, d)
end




