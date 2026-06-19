"""
    hotelling(X, y::Vector{String}; digits::Int = 5)
Two-sample Hotelling's T-squared test.
* `X` : X-data (n, p).
* `y` : Univariate categorical variable (2-class membership) (n). Must be a `Vector{String}`.
Keyword arguments:
* `digits` : Nb. digits for the outputs.

## Examples 
```julia
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/reaction_bertinetto.jld2")
@load db dat
@names dat
datf = dat.datf
n = nro(datf)
tab(datf; group = [:temp, :catal])  # balanced design
##
Y = datf[:, [:y1, :y2]]
fact = datf.catal
tab(fact)

hotelling(Y, fact)

test = :wilks 
#test = :pillai
#test = :hotelling
#test = :roy
B = matB(Y, fact, pweight(ones(n))).B
W = matW(Y, fact, pweight(ones(n))).W
@show wilks(B * inv(W))[test]
manova(Y, @formula(0 ~ catal), datf; test)
```
"""
function hotelling(X, y::Vector{String}; digits::Int = 5)
    n, p = size(X)
    res = matWc(X, y)
    ni = res.ni 
    xmeans = aggmean(X, y).X
    xdif = xmeans[2, :] - xmeans[1, :]
    t2 = ni[1] * ni[2] / n * dot(xdif, inv(res.W) * xdif)
    F = (n - p - 1) / ((n - 2) * p) * t2
    dfnum = p
    dfden = n - p - 1
    d = Distributions.FDist(dfnum, dfden)
    pval = Distributions.ccdf(d, F)
    res = round.((t2, F, dfnum, dfden, pval); digits)
    (; zip((:t2, :F, :dfnum, :dfden, :pval), res)...)
end
