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
```
"""
function asca(X, f::StatsModels.FormulaTerm, dat::DataFrame)
    res = decompx(X, f, dat)
    explvarx = summary(res)
    df = res.df.dffit[2:end]
    ## PCAs on fitted values
    nterm = length(res.fit) - 1
    fitm_pca = list(Jchemo.Pca, nterm)
    for i in 1:nterm
        fitm_pca[i] = pcasvd(res.fit[i + 1]; nlv = df[i])
    end
    (fitm_pca = fitm_pca, explvarx)
end

