struct Occstah
    d
    res_stah
    e_cdf
    cutoff::Real
end

"""
    occstah(X; a = 2000, scal = true, 
        typc = "mad", cri = 3, alpha = .05)
One-class classification using the Stahel-Donoho outlierness measure.

* `X` : X-data.
* `a` : Nb. dimensions simulated for the projection-pursuit method.
* `scal` : Boolean. If `true`, matrix `X` is centred (by median) 
    and scaled (by MAD) before computing the outlierness.
* `typc` : Type of cutoff ("mad" or "q"). See below.
* `cri` : When `typc = "mad"`, constant used for computing the 
    cutoff detecting extreme values.
* `alpha` : When `typc = "q"`, risk-I level used for computing the cutoff 
    detecting extreme values.

In this method, the "outlierness measure" `d` of a given observation
is the Stahel-Donoho outlierness (see `?stah`).

See `?occsd` for the cutoff computation (the same principle is applied). 

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
Ttest = Jchemo.transform(fm, zXtest)
T = vcat(Ttrain, Ttest)
group = vcat(repeat(["0-Train"], ntrain), repeat(["1-Test"], ntest))
i = 1
plotxy(T[:, i], T[:, i + 1], group;
    xlabel = string("PC", i), ylabel = string("PC", i + 1)).f

#### End data

fm = occstah(zXtrain) ;
fm.d
hist(fm.d.dstand; bins = 50)

res = Jchemo.predict(fm, zXtest) ;
res.d
res.pred
tab(res.pred)

d1 = fm.d.dstand
d2 = res.d.dstand
d = vcat(d1, d2)
f, ax = plotxy(1:ntot, d, group)
hlines!(ax, 1)
f
```
""" 
function occstah(X; a = 2000, scal = true, 
        typc = "mad", cri = 3, alpha = .05) 
    res = Jchemo.stah(X, a; scal = scal)
    d = res.d
    #d2 = d.^2 
    #mu = median(d2)
    #s2 = mad(d2)^2
    #nu = 2 * mu^2 / s2
    #g = mu / nu
    #dist = Distributions.Chisq(nu)
    #pval = Distributions.ccdf.(dist, d2 / g)
    #typc == "par" ? cutoff = sqrt(g * quantile(dist, 1 - alpha)) : nothing
    #typc == "npar" ? cutoff = median(d) + cri * mad(d) : nothing  
    typc == "mad" ? cutoff = median(d) + cri * mad(d) : nothing
    typc == "q" ? cutoff = quantile(d, 1 - alpha) : nothing
    e_cdf = StatsBase.ecdf(d)
    pval = 1 .- e_cdf(d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = pval)
    Occstah(d, res, e_cdf, cutoff)
end

"""
    predict(object::Occstah, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occstah, X)
    zX = copy(ensure_mat(X))
    m = nro(zX)
    res = object.res_stah
    center!(zX, res.mu_scal)
    scale!(zX, res.s_scal)
    T = zX * res.P
    center!(T, res.mu)
    scale!(T, res.s)
    T .= abs.(T)
    d = similar(T, m)
    @inbounds for i = 1:m
        d[i] = maximum(vrow(T, i))
    end
    #pval = Distributions.ccdf.(object.dist, d.^2 / object.g)
    pval = 1 .- object.e_cdf(d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = pval)
    pred = reshape(Int64.(d.dstand .> 1), m, 1)
    (pred = pred, d)
end


