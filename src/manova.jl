"""
    manova(Y, f::StatsModels.FormulaTerm, dat::DataFrame; test = :pillai, digits = 4)
MANOVA.
* `Y` : Y-data (n, p) representing the response variables.
* `f` : A formula that defines the tested factor(s). See the syntax in the examples below.
* `dat` (n, q) : Dataframe containing the factor(s) specified in `f`. 
Keyword arguments:
* `test` : Type of statistic used for the test. Possible values are: `:wilks`, `:pillai` (default), 
    `:hotelling`, or `:roy`.
* `digits` : Nb. digits for the outputs.

The function returns approximated F tests for one of the the following statistics (argument `test`):
* Wilks’ lambda
* Pillai’s trace
* Hotelling-Lawley trace
* Roy’s maximum root

## References
https://documentation.sas.com/doc/en/statug/15.2/statug_introreg_sect038.htm#statug_introreg001918

## Examples 
```julia
```
"""
function manova(Y, f::StatsModels.FormulaTerm, dat::DataFrame; test = :pillai, digits = 4)
    @assert in([:wilks; :pillai; :hotelling; :roy])(test) "Wrong value for argument 'test'." 
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
    dfden = list(Int, length(L))
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
        if test == :wilks
            c = p^2 + q^2 - 5
            if c > 0
                t = sqrt(((p * q)^2 - 4) / c)
            else
                t = 1
            end
            r = nu - (p - q + 1) / 2
            u = (p * q - 2) / 4 
            a = r * t - 2 * u
            b = p * q
            F[i] = a / b * (1 - val[i]^(1 / t)) / val[i]^(1 / t) 
            dfnum[i] = b 
            dfden[i] = a
        elseif test == :pillai
            a = 2 * n + s + 1
            b = 2 * m + s + 1
            F[i] = a / b * val[i] / (s - val[i]) 
            dfnum[i] = s * b
            dfden[i] = s * a
        elseif test == :hotelling
            F[i] = 2 * (s * n + 1) * val[i] / (s^2 * (2 * m + s + 1))
            dfnum[i] = s * (2 * m + s + 1)
            dfden[i] = 2 * (s * n + 1)
            ## This version of the function follows the R's choice to not use
            ## the following variant when n > 0 (used in SAS, see ref):
            ## b = (p + 2 * n) * (q + 2 * n) / (2 * (2 * n + 1) * (n - 1))
            ## c = (2 + (p * q + 2) / (b - 1)) / (2 * n)
            ## F[i] = (val[i] / c) * ((4 + (p * q + 2) / (b - 1)) / (p * q))
            ## dfnum[i] = p * q
            ## dfden[i] = 4 + (p * q + 2) / (b - 1)
            ## End
        elseif test == :roy
            r = max(p, q)
            a = nu - r + q
            F[i] = a / r * val[i]
            dfnum[i] = r
            dfden[i] = a
        end
        d = Distributions.FDist(dfnum[i], dfden[i])
        pval[i] = Distributions.ccdf(d, F[i])
    end        
    pval .= round.(pval; digits)
    nam = collect(@names res.fit)[2:end]
    df = res.df.dffit[2:end]
    DataFrame(:term => nam, :df => df, test => val, :approxF => F, :dfnum => dfnum, :dfden => dfden, :pval => pval)
end 
