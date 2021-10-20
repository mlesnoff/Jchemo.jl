"""
    aov1(x, Y)
    Univariate anova test.
* `x` : Univariate categorical X-data.
* `Y` : Y-data.
""" 
function aov1(x, Y)
    Y = ensure_mat(Y)
    n = length(x)
    q = size(Y, 2) 
    A = length(unique(x))
    Xdummy = dummy(x).Y
    zY = center(Y, colmeans(Y))
    fm = mlr(Xdummy, zY) ;
    pred = predict(fm, Xdummy).pred
    SSF = sum((pred.^2), dims = 1)   # = colvars(pred) * n
    SSR = ssr(pred, zY)
    df_fact = A - 1 
    df_res = n - A
    MSF = SSF / df_fact
    MSR = SSR / df_res
    F = MSF ./ MSR
    d = Distributions.FDist(df_fact, df_res)
    pval = Distributions.ccdf(d, F)
    (SSF = SSF, SSR = SSR, df_fact = df_fact, df_res = df_res, F = F, pval = pval)
end





