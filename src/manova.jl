"""
    manova(; digits = 4)
MANOVA with approximated F tests.
* `b` : Vector (p) of the coefficients of the model.
* `L` : Matrix (m, p) such as `L` * `b` gives the linear combination(s) of the coefficients 
    to be tested.
* `varb` : Variance-covariance matrix (p, p) of `b`.
Keyword arguments:
* `digits` : Nb. digits for the outputs.

The function tests hypothesis H0: `L` * `b` = `h0`, with either 
* a Chi-squared Wald test (with dfs = m)
* or, if `defden` is given, a F test (with dfs {m, `defden`}).

Both tests assume that `b` is Gaussian.  Compared to the F test, the Wald test neglects the uncertainty 
affecting the estimate of the dispersion parameter of the model (e.g., 'sigma2' in MLRs). 

## References
https://documentation.sas.com/doc/en/statug/15.2/statug_introreg_sect038.htm#statug_introreg001918

## Examples 
```julia
```
"""
function manova(Y, f::StatsModels.FormulaTerm, dat::DataFrame; test = :pillai, digits = 4)
    Y = ensure_mat(Y)
    Q = eltype(Y)
    res = decompx(Y, f, dat)
    B = res.mat.B
    D = res.mat.D
    DtD = D' * D      
    L = res.mat.L
    Yc = fcenter(Y, res.xmeans)
    E = Yc' * Yc - B' * DtD * B 
    invE = inv(E)
    nu = res.df.dfr
    val = list(Q, length(L))
    F = copy(val)
    dfnum = list(Int, length(L))
    defden = list(Int, length(L))
    pval = copy(val)
    for i in eachindex(L)
        LB = L[i] * B
        A = L[i] * inv(DtD) * L[i]'
        H = LB' * inv(A) * LB
        val[i] = wilks(invE * H)[test]
        ##
        p = Int(rank(E + H))
        q = Int(rank(A))
        s = min(p, q)
        m = (abs(p - q) - 1) / 2
        n = (nu - p - 1) / 2
        if test == :pillai
            F[i] = (2 * n + s + 1) / (2 * m + s + 1) * val[i] / (s - val[i]) 
            dfnum[i] = s * (2 * m + s + 1)
            defden[i] = s * (2 * n + s + 1)
        end
        d = Distributions.FDist(dfnum[i], defden[i])
        pval[i] = Distributions.ccdf(d, F[i])
    end
    pval .= round.(pval; digits)
    nam = collect(@names res.fit)[2:end]
    DataFrame(:term => nam, test => val, :approxF => F, :dfnum => dfnum, :defden => defden, 
        :pval => pval)
end 
