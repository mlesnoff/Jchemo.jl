"""
    aov1(X::AbstMatVec{Q}, y::Vector{String}) where Q <: Float 
    One-factor ANOVA test.
* `X` : X-data (n, p) whose columns are tested (independently).
* `y` : A categorical variable (class membership) (n). Must be a `Vector{String}`.

## Examples
```julia
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
@names dat
@head dat.X
X = Matrix(dat.X[:, 1:4])
y = dat.X[:, 5]
tab(y) 

res = aov1(X, y) ;
@names res
res.SSF
res.SSR 
res.F 
res.pval
```
""" 
function aov1(X::AbstMatVec{Q}, y::Vector{String}) where Q <: Float 
    X = ensure_mat(X)
    tabx = tab(y)
    lev = tabx.keys
    ni = tabx.vals
    nlev = length(lev)
    Ydummy = dummy(Q, y).Y
    Xc = fcenter(X, colmean(X))
    fitm = mlr(Ydummy, Xc)
    pred = predict(fitm, Ydummy).pred
    SSF = sum((pred.^2); dims = 1)   # return a matrix
    SSR = ssr(pred, Xc)              # return a matrix
    df_fact = nlev - 1 
    df_res = nro(X) - nlev
    MSF = SSF / df_fact
    MSR = SSR / df_res
    F = MSF ./ MSR
    d = Distributions.FDist(df_fact, df_res)
    pval = Distributions.ccdf(d, F)
    (SSF = SSF, SSR, df_fact, df_res, F, pval, ni, lev)
end
