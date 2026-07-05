"""
    decompx(X::AbstMatVec{Q}, f::StatsModels.FormulaTerm, datf::DataFrame) where Q <: Float
Decomposition of a matrix by orthogonal projection on experimental factors.
* `X` : A matrix (n, p) or vector (n) to decompose.
* `f` : A formula that defines the factor(s) on which is(are) done the decomposition.
    See the syntax in the examples below.
* `datf` (n, q) : Dataframe containing the factor(s) specified in `f`. 

## References
Bertinetto, C., Engel, J., Jansen, J., 2020. ANOVA simultaneous component analysis: A tutorial review. 
Analytica Chimica Acta: X 6, 100061. https://doi.org/10.1016/j.acax.2020.100061

Doledec, S., Chessel, D., 1987. Rythmes saisonniers et composantes stationnelles en milieu aquatique. 
I: Description d’un plan d’observation complet par projection de variables. 
Acta oecol., Oecol. gen 8, 403–426.

Smilde, A.K., Jansen, J.J., Hoefsloot, H.C.J., Lamers, R.-J.A.N., van der Greef, J., Timmerman, M.E., 2005. 
ANOVA-simultaneous component analysis (ASCA): a new tool for analyzing designed metabolomics data. 
Bioinformatics 21, 3043–3048. https://doi.org/10.1093/bioinformatics/bti476

Smilde, A.K., Marini, F., Westerhuis, J.A., Liland, K.H. (Eds.), 2025. Analysis of variance 
for high-dimensional data: applications in life, food and chemical sciences. Wiley, Hoboken, NJ.

## Examples 
```julia
#### Example of decomposition reported in Bertinetto et al. act chim. acta 2020 (section 2.1).

using Jchemo, JchemoData, JLD2, StatsModels
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/reaction_bertinetto.jld2")
@load db dat
@names dat
datf = dat.datf
n = nro(datf)
tab(datf; group = [:temp, :catal])  # balanced design
##
Y = Matrix(datf[:, [:y1, :y2]])
aggstat(datf; sel = [:y1, :y2], group = [:temp])
aggstat(datf; sel = [:y1, :y2], group = [:catal])
res = aggstat(datf; sel = [:y1, :y2], group = [:temp, :catal])

f = @formula(0 ~ temp + catal + temp & catal)
#f = @formula(0 ~ temp + catal)
#f = @formula(0 ~ temp)

d = decompx(Y, f, datf) ;
@names d

## Fitted values of the effects
@names d.fit
nam = :temp
#nam = Symbol("temp & catal")
d.fit[nam]

## Residuals
d.R

## Rebuild Y
@head Y
@head reduce(+, d.fit) + d.R

## Explained variances
summary(d)
summary(d; corrected = false)

## Permutation tests
res = permut(d; rep = 1000) ;
res.explvarx
res.valref
@head res.val
```
"""
function decompx(X::AbstMatVec{Q}, f::StatsModels.FormulaTerm, datf::DataFrame) where Q <: Float
    n = nro(X)
    xmeans = colmean(X)
    Xc = fcenter(X, xmeans)
    Xm = ones(Q, n) * xmeans'
    ## Contrasts
    contr = EffectsCoding()   # sum-to-zero
    term_princ = Symbol.(terms(f.rhs))     # [2:end]
    nterm_princ = length(term_princ)
    tupl = (; zip(term_princ, fill(contr, nterm_princ))...)
    nam = @names tupl
    contrasts = Dict{Symbol, EffectsCoding}()
    for i in term_princ
        contrasts[i] = contr
    end 
    ## D, B
    mf = ModelFrame(f, datf; contrasts)
    fs = apply_schema(f, mf.schema)
    resp, D = modelcols(fs, datf) ;   # no intercept
    ## If argument 'perm' is added in the future
    ## if perm
    ##     D .= D[randperm(n), :]
    ## end
    ## End
    B = inv(D' * D) * D' * Xc        # no intercept
    dfm = nco(D) + 1                 # include intercept
    ## Assign terms
    term_rhs = fs.rhs.terms
    nterm_rhs = length(term_rhs) 
    assign = StatsModels.asgn(term_rhs)
    ## Fit (including Intercept term) and R
    C = list(Matrix{Int}, nterm_rhs)
    L = list(Matrix{Int}, nterm_rhs)    
    M = list(Matrix{Int}, nterm_rhs)
    namfit = vcat("Intercept", collect(string.(term_rhs)))
    fit = list(Matrix{Q}, nterm_rhs + 1)
    fit[1] = copy(Xm)
    c = zeros(dfm - 1)
    for i in eachindex(term_rhs)
        c[assign .== i] .= 1
        C[i] = diagm(c)
        c .= zeros(dfm - 1)
        L[i] = C[i][assign .== i, :]
        M[i] = D * C[i]
        fit[i + 1] = M[i] * B
    end
    R = Xc - D * B
    ss = (sst = frob2(X), ssfit =  frob2.(fit), ssr = frob2(R))
    dffit = vcat(1, tab(assign).vals)
    dfr = n - dfm
    df = (dffit = dffit, dfr, dftot = n)
    mat = (B = B, D, C, L, M)
    fit = (; zip(Symbol.(namfit), fit)...)
    Decompx(fit, R, mat, ss, df, f, assign, datf, xmeans)
end

"""
    summary(object::Decompx; corrected::Bool = true, digits::Int = 4)
Summarize the fitted model.
* `object` : Object of class `Decompx` to summarize.
Keyword arguments:
* `corrected` : Whether to correct for the intercept term.
* `digits` : Nb. digits for the outputs.
""" 
function Base.summary(object::Decompx; corrected::Bool = true, digits::Int = 4)
    namfit = @names object.fit
    ssfit = object.ss.ssfit
    ssr = object.ss.ssr
    dffit = object.df.dffit
    dfr = object.df.dfr
    if corrected
        namfit = namfit[2:end]
        ssfit = ssfit[2:end]
        dffit = dffit[2:end]
    end
    sst = sum(ssfit) + ssr
    pvar = vcat(ssfit, ssr) / sst
    cumpvar = cumsum(pvar)
    nam = [collect(namfit); :Residuals]
    res = DataFrame(term = nam, df = vcat(dffit, dfr), ss = vcat(ssfit, ssr), pvar = pvar, cumpvar = cumpvar)
    u = [:ss, :pvar, :cumpvar]
    res[:, u] = round.(res[:, u]; digits)
    res
end

"""
    permut(object::Decompx; rep::Int = 1000, digits::Int = 4)
Permutation test of effects after decomposition of a matrix by experimental factors.
* `object` : Object of type `Decompx` (output of function `decompx`).
* `datf` : Dataframe containing the factor(s) specified in `object.f`.
Keyword arguments:
* `rep` : Number of unrestricted permutations (of X rows) for testing the significance of the effects.
* `digits` : Nb. digits for the outputs.

The function performs unrestricted permutations of the X-data rows (X being rebuilt from `object`) and compares the 
distribution of SS(effect) / SSR (computed from the set of `rep` permutations) to the reference value.

See function `decompx` for examples. 

## References
Manly, B.F., 2007. Randomization, bootstrap and Monte Carlo methods in biology, 3rd ed. Chapman & Hall/CRC, Boca Raton.

Smilde, A.K., Marini, F., Westerhuis, J.A., Liland, K.H. (Eds.), 2025. Analysis of variance 
for high-dimensional data: applications in life, food and chemical sciences. Wiley, Hoboken, NJ.
""" 
function permut(object::Decompx; rep::Int = 1000, digits::Int = 4)
    X = reduce(+, object.fit) + object.R
    n = nro(X)
    valref = object.ss.ssfit[2:end] / object.ss.ssr
    nterm = length(valref)
    val = similar(valref, rep, nterm)
    s = list(Int, n)
    for i in axes(val, 1)
        s .= randperm(n)
        res = decompx(vrow(X, s), object.f, object.datf)
        val[i, :] .= res.ss.ssfit[2:end] / res.ss.ssr
    end
    pv = [Jchemo.pval(val[:, i], valref[i]) for i in axes(val, 2)]
    explvarx = summary(object; digits)
    explvarx.pval = round.(vcat(pv, NaN); digits)
    (explvarx = explvarx, valref, val)
end



