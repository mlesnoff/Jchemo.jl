"""
    aov1(x, Y)
    One-factor ANOVA test.
* `x` : Univariate categorical (factor) data (n).
* `Y` : Y-data (n, q).

## Examples
```julia
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
pnames(dat)
@head dat.X
x = dat.X[:, 5]
Y = dat.X[:, 1:4]
tab(x) 

res = aov1(x, Y) ;
pnames(res)
res.SSF
res.SSR 
res.F 
res.pval
```
""" 
function aov1(x, Y)
    Y = ensure_mat(Y)
    n = length(x)
    tabx = tab(x)
    lev = tabx.keys
    ni = tabx.vals
    nlev = length(lev)
    Xdummy = dummy(x).Y
    Yc = fcenter(Y, colmean(Y))
    fitm = mlr(convert.(Float64, Xdummy), Yc)
    pred = predict(fitm, Xdummy).pred
    SSF = sum((pred.^2); dims = 1)   # = colvar(pred) * n
    SSR = ssr(pred, Yc)
    df_fact = nlev - 1 
    df_res = n - nlev
    MSF = SSF / df_fact
    MSR = SSR / df_res
    F = MSF ./ MSR
    d = Distributions.FDist(df_fact, df_res)
    pval = Distributions.ccdf(d, F)
    (SSF = SSF, SSR, df_fact, df_res, F, pval, lev, ni)
end
