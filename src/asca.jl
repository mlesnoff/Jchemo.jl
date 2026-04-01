"""
    asca(X, f::StatsModels.FormulaTerm, dat::DataFrame)
ANOVA Simultaneous Component Analysis (ASCA).
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

f = @formula(0 ~ temp + catal + temp & catal)
#f = @formula(0 ~ temp + catal)
#f = @formula(0 ~ temp)
#f = @formula(0 ~ catal)

d = decompx(Y, f, datf) ; # decompsition of Y according to the effects in f
res = asca(d) ;
@names res
res.df

fitm = res.fitm ; # Pca models on the effect matrices
@names fitm
nam = :temp
#nam = Symbol("temp & catal")
@head fitm[nam].T
fitm[nam].V
```
"""
function asca(object::Decompx)
    nterm = length(object.fit) - 1
    namterm = (@names object.fit)[2:end]
    df = object.df.dffit[2:end]
    fitm = list(Jchemo.Pca, nterm)
    for i in 1:nterm
        fitm[i] = pcasvd(object.fit[i + 1]; nlv = df[i])
    end
    fitm = (; zip(namterm, fitm)...)
    (fitm = fitm, df)
end




