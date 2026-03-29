struct Decompx2
    fit::NamedTuple
    R::Matrix
    mat::NamedTuple
    ss::NamedTuple
    df::NamedTuple
    assign::Vector{Int}
    xmeans::Vector 
end

"""
    decompx(X, f::StatsModels.FormulaTerm, dat::DataFrame)
Decomposition of a matrix by orthogonal projection on factors.
* `X` :  X-data (n, p) to decompose.
* `f` : A formula that defines the factor(s) on which is(are) done the decomposition.
    See the syntax in the examples below.
* `dat` (n, q) : Dataframe containing the factor(s) specified in `f`. 

## References
Bertinetto, C., Engel, J., Jansen, J., 2020. ANOVA simultaneous component analysis: A tutorial review. 
Analytica Chimica Acta: X 6, 100061. https://doi.org/10.1016/j.acax.2020.100061

Doledec, S., Chessel, D., 1987. Rythmes saisonniers et composantes stationnelles en milieu aquatique. 
I: Description d’un plan d’observation complet par projection de variables. 
Acta oecol., Oecol. gen 8, 403–426.

Smilde, A.K., Jansen, J.J., Hoefsloot, H.C.J., Lamers, R.-J.A.N., van der Greef, J., Timmerman, M.E., 2005. 
ANOVA-simultaneous component analysis (ASCA): a new tool for analyzing designed metabolomics data. 
Bioinformatics 21, 3043–3048. https://doi.org/10.1093/bioinformatics/bti476

## Examples 
```julia
## Example of decomposition reported in Bertinetto et al. act chim. acta 2020 (section 2).
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/reaction_bertinetto.jld2")
@load db dat
@names dat
datf = dat.datf
n = nro(datf)
tab(datf; group = [:temp, :catal])  # balanced design

Y = datf[:, [:y1, :y2]]
aggstat(datf; sel = [:y1, :y2], group = :temp)
aggstat(datf; sel = [:y1, :y2], group = :catal)
aggstat(datf; sel = [:y1, :y2], group = [:temp, :catal])

f = @formula(0 ~ temp + catal + temp & catal)
#f = @formula(0 ~ temp + catal)
#f = @formula(0 ~ temp)
#f = @formula(0 ~ catal)

res = decompx(Y, f, datf) ;
@names res
res.namfit 
res.fit
res.fit[2]
res.R
res.ss
res.df 

@head Y
@head reduce(+, res.fit) + R
```
"""
function decompx(X, f::StatsModels.FormulaTerm, dat::DataFrame)
    X = ensure_mat(X)
    Q = eltype(X)
    n = nro(X)
    xmeans = colmean(X)
    Xc = fcenter(X, xmeans)
    Xm = ones(Q, n) * xmeans'
    ## Contrasts
    contr = EffectsCoding()   # sum-to-zero
    term_princ = Symbol.(terms(f.rhs))     # [2:end]
    nterm_princ = length(term_princ)
    tupl = (; zip(term_princ, repeat([contr], nterm_princ))...)
    nam = @names tupl
    contrasts = Dict{Symbol, EffectsCoding}()
    for i in term_princ
        contrasts[i] = contr
    end 
    ## D, B
    mf = ModelFrame(f, dat; contrasts)
    fs = apply_schema(f, mf.schema)
    resp, D = modelcols(fs, dat) ;   # no intercept
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
    Decompx2(fit, R, mat, ss, df, assign, xmeans)
end

"""
    summary(object::Decompx2; corrected = true, digits = 4)
Summarize the fitted model.
* `object` : The fitted model.
* `corrected` : Whether to correct for the intercept term.
* `digits` : Nb. digits for the outputs.
""" 
function Base.summary(object::Decompx2; corrected = true, digits = 4)
    ssfit = object.ss.ssfit
    namfit = @names object.fit
    ssr = object.ss.ssr
    if corrected
        ssfit = ssfit[2:end]
        namfit = namfit[2:end]
    end
    sst = sum(ssfit) + ssr
    pvar = vcat(ssfit, ssr) / sst
    cumpvar = cumsum(pvar)
    nam = [collect(namfit); :Residuals]
    res = DataFrame(term = nam, ss = vcat(ssfit, ssr), pvar = pvar, cumpvar = cumpvar)
    u = [:ss, :pvar, :cumpvar]
    res[:, u] = round.(res[:, u]; digits)
    res
end


