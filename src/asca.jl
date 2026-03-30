"""
    asca(X, f::StatsModels.FormulaTerm, dat::DataFrame)
ANOVA Simultaneous Component Analysis (ASCA).
* `X` :  X-data (n, p) to decompose.
* `f` : A formula that defines the factor(s) on which is(are) done the decomposition.
    See the syntax in the examples below.
* `dat` (n, q) : Dataframe containing the factor(s) specified in `f`. 
Keyword arguments:
* `rep` : Number of unrestricted permutations (of X rows) for testing the significance of the effects.
    If `rep` = 0 (default), no permutation is done (no test).

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
Y = Matrix(Y)

f = @formula(0 ~ temp + catal + temp & catal)
#f = @formula(0 ~ temp + catal)
#f = @formula(0 ~ temp)
#f = @formula(0 ~ catal)

res = asca(Y, f, datf) ;
@names res

## Explained variance of X by the effects
res.explvarx 

## PCAs on the fitted values res.decomp.fit[nam]
fitm_pca = res.fitm_pca
@names fitm_pca
nam = :temp
#nam = Symbol("temp & catal")
@head fitm_pca[nam].T
fitm_pca[nam].V

## Permutation tests of the effects
res = asca(Y, f, datf; rep = 1000) ;
res.explvarx 
```
"""
function asca(X, f::StatsModels.FormulaTerm, dat::DataFrame; rep = 0)
    res = decompx(X, f, dat)
    explvarx = summary(res)
    ## PCAs on fitted values
    nterm = length(res.fit) - 1
    namterm = (@names res.fit)[2:end]
    df = res.df.dffit[2:end]
    fitm_pca = list(Jchemo.Pca, nterm)
    for i in 1:nterm
        fitm_pca[i] = pcasvd(res.fit[i + 1]; nlv = df[i])
    end
    fitm_pca = (; zip(namterm, fitm_pca)...)
    ## Permutation tests
    ## Perm X rows and compute SS / SSR
    if rep > 0
        n = nro(dat)
        valc = res.ss.ssfit[2:end] / res.ss.ssr
        nterm = length(valc)
        val = similar(valc, rep, nterm)
        s = list(Int, n)
        for i = 1:rep
            s .= randperm(n)
            zX = vrow(X, s)
            vres = decompx(zX, f, dat)
            val[i, :] .= vres.ss.ssfit[2:end] / vres.ss.ssr
        end
        pv = [Jchemo.pval(val[:, i], valc[i]) for i in axes(val, 2)]
        explvarx.pval = vcat(pv, NaN)
    end  
    ##
    (fitm_pca = fitm_pca, explvarx, decomp = res)
end

