"""
    hotelling(X, y; digits = 5)
Two-sample Hotelling's T-squared test.
* `X` : X-data (n, p).
* `y` : A vector (n) defining the 2-class membership.
Keyword arguments:
* `digits` : Nb. digits for the outputs.

## Examples 
```julia
```
"""
function hotelling(X, y; digits = 5)
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
