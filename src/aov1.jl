"""
    aov1(x, Y)
    Univariate anova test.
* `x` : Univariate categorical X-data.
* `Y` : Y-data.

## Examples
```julia
n = 100 ; p = 5
x = rand(1:3, n)
Y = randn(n, p) 

res = aov1(x, Y)
pnames(res)
res.SSF
res.SSR 
res.F 
res.pval
```
""" 
function aov1(x, Y)
    Y = ensure_mat(Y)
    Q = eltype(Y)
    n = length(x)
    ztab = tab(x)
    lev = ztab.keys
    ni = ztab.vals
    nlev = length(lev)
    Xdummy = dummy(x, Q).Y
    zY = fcenter(Y, colmean(Y))
    fm = mlr(Xdummy, zY) ;
    pred = predict(fm, Xdummy).pred
    SSF = sum((pred.^2), dims = 1)   # = colvar(pred) * n
    SSR = ssr(pred, zY)
    df_fact = nlev - 1 
    df_res = n - nlev
    MSF = SSF / df_fact
    MSR = SSR / df_res
    F = MSF ./ MSR
    d = Distributions.FDist(df_fact, df_res)
    pval = Distributions.ccdf(d, F)
    (SSF = SSF, SSR, df_fact, df_res, F, pval,
        lev, ni)
end
