"""
    manova(Y, f::StatsModels.FormulaTerm, dat::DataFrame; test = :pillai, digits = 4)
MANOVA.
* `Y` : Response variables (n, p).
* `f` : A formula that defines the tested factor(s). See the syntax in the examples below.
* `dat` (n, q) : Dataframe containing the factor(s) specified in `f`. 
Keyword arguments:
* `test` : Type of test statistic. Possible values are: `:wilks`, `:pillai` (default), `:hotelling`, or `:roy`.
* `lb` : Positive constant for regularization.
* `digits` : Nb. digits for the outputs.

The function returns approximated F tests for one of the the following statistics (argument `test`):
* Wilks’ lambda
* Pillai’s trace
* Hotelling-Lawley trace
* Roy’s maximum root

If `lb` > 0, the function performs a ridge regularization by adding `lb` to the diagonal of R'R, where 
R (n, p) is the residual matrix.

## References
https://documentation.sas.com/doc/en/statug/15.2/statug_introreg_sect038.htm#statug_introreg001918

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
aggstat(datf; sel = [:y1, :y2], group = :temp)
aggstat(datf; sel = [:y1, :y2], group = :catal)
res = aggstat(datf; sel = [:y1, :y2], group = [:temp, :catal])

f = @formula(0 ~ temp + catal + temp & catal)
#f = @formula(0 ~ temp + catal)
#f = @formula(0 ~ temp)
#f = @formula(0 ~ catal)

manova(Y, f, datf)

manova(Y, f, datf; test = :wilks)
```
"""
function manova(Y, f::StatsModels.FormulaTerm, dat::DataFrame; test = :pillai, lb = 0, digits = 4)
    @assert in([:wilks; :pillai; :hotelling; :roy])(test) "Wrong value for argument 'test'." 
    Y = ensure_mat(Y)
    Q = eltype(Y)
    res = decompx(Y, f, dat)
    B = res.mat.B
    D = res.mat.D
    DtD = D' * D      
    L = res.mat.L
    Yc = fcenter(Y, res.xmeans)
    RtR = Yc' * Yc - B' * DtD * B 
    if lb > 0
        RtR .+= lb * I(nco(Y))
    end
    invRtR = inv(RtR)
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
        val[i] = wilks(invRtR * H)[test]
        ##
        p = Int(rank(RtR + H))
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
            ## Approximation McKeon, 1974
            F[i] = 2 * (s * n + 1) * val[i] / (s^2 * (2 * m + s + 1))
            dfnum[i] = s * (2 * m + s + 1)
            dfden[i] = 2 * (s * n + 1)
            ## The actual version of the function follows the R's choice,
            ## not the SAS' one when n > 0 (see ref):
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
    ss = round.(res.ss.ssfit[2:end]; digits)
    df = res.df.dffit[2:end]
    DataFrame(:term => nam, :ss => ss, :df => df, test => val, :approxF => F, :dfnum => dfnum, :dfden => dfden, :pval => pval)
end 
