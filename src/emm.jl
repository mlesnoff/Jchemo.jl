"""
    emm(fitm::StatsModels.TableRegressionModel, f::StatsModels.FormulaTerm, dat::DataFrame)
Estimated marginal means (EMMs).
* `fitm` : A model fitted with package GLM.
* `f` : A formula that defines the model factor(s) on which is(are) computed the EMMs.
    Must be additive (interaction terms not allowed). See the syntax in the example below.
* `dat` : DataFrame on which `fitm` has been estimated. 

The function computes estimated marginal means (EMMs) (Searle et al 1980) from a model fitted with package GLM 
(https://github.com/JuliaStats/GLM.jl). EMMs are unweighted marginal means of the cell means predicted by
the given model, a.k.a 'least-squares means' in the SAS GLM terminology. When the model contains 
continuous variables, the cell means are predicted at the mean of these continuous variables.

## References
- Searle, S.R., Speed, F.M., Milliken, G.A., 1980. Population Marginal Means in the Linear Model: 
    An Alternative to Least Squares Means. The American Statistician 34, 216–221. 
    https://doi.org/10.2307/2684063

## Examples 
```julia
using DataFrames
using GLM

using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/flour_splus6.jld2")
@load db dat
@names dat
datf = dat.datf   # uncomplete design without replication
n = nro(datf)
datf.x = rand(n)

## Initial model
f_fitm = @formula(y ~ 1 + flour + fat * surfact)
fitm = lm(f_fitm, datf)

## Define factor(s) on which is(are) computed EMMs
f = @formula(0 ~ flour)
#f = @formula(0 ~ fat)
#f = @formula(0 ~ surfact)
#f = @formula(0 ~ fat + surfact)

res = emm(fitm, f, datf) ;
@names res
res.datemm

## Results reported from Manual Splus6, p. 632-633
## EMMs fo model 'flour + fat * surfact'
##
## Tables of adjusted means
##   Grand mean
##     6.633281
##  se 0.084599
## Flour
##       1      2     3      4
##    7.3020 5.7073 6.9815 6.5423
## se 0.1995 0.1467 0.1621 0.1785
## Fat
##       1     2      3
##    5.8502 6.5771 7.4725
## se 0.1365 0.1477 0.1565
## Surfactant
##       1      2      3
##    6.3960 6.5999 6.9039
## se 0.1502 0.1432 0.1473
## Fat:Surfactant
## Dim 1 : Fat
## Dim 2 : Surfactant
##      1       2     3
## 1  5.5364 5.8913 6.1229
## se 0.2404 0.2392 0.2414
## 2  7.0229 6.7085 6.0000
## se 0.2414 0.3006 0.2034
## 3  6.6286 7.2000 8.5889
## se 0.3007 0.2034 0.3001
```
"""
function emm(fitm::StatsModels.TableRegressionModel, f::StatsModels.FormulaTerm, dat::DataFrame)
    f_fitm = formula(fitm)       # = fitm.mf.f    
    b = StatsAPI.coef(fitm)
    varb = StatsAPI.vcov(fitm)   # to do: check if correct (delta-method) for glims
    ## Build the full table corresponding to model fitm
    nam = string.(@names dat)
    u = in(termnames(f_fitm.rhs)).(nam)
    keys = nam[u]
    vdat = dat[:, keys]
    for i in axes(vdat, 2)
        if isa(vdat[1, i], Real)
            vdat[:, i] .= meanv(vdat[:, i])
        end
    end
    values = ntuple(i -> mlev(vdat[:, i]), length(keys))
    tupl = (; zip(Symbol.(nam[u]), values)...)   # better than : NamedTuple{keys}(values)
    datmu = Jchemo.expand_grid_tupl(tupl)
    ## Estimate the mean 'mu' for each cell of the full table 
    namterm = termnames(f_fitm.rhs)
    namterm[namterm .== "(Intercept)"] .= "1"
    fst = string("0~", join(namterm, "+"))
    ftmp = eval(Meta.parse("@formula($fst)"))
    mf = ModelFrame(ftmp, datmu)
    fs = apply_schema(ftmp, mf.schema)
    resp, D = modelcols(fs, datmu) ;
    D
    mu = D * b
    varmu = D * varb * D'
    datmu.pred = mu
    datmu.se = sqrt.(diag(varmu))
    datmu
    ## Compute EMMs
    nam = termnames(f)[2]
    isa(nam, Vector) ? nothing : nam = [nam]
    nam
    datmu[:, nam]
    v = [Vector(datmu[i, nam]) for i in axes(datmu, 1)]    
    fact = join.(v, ["-"])
    V = unique(datmu[:, nam])
    L = Float64.(dummy(fact).Y')
    v = 1 ./ rowsum(L)
    fweightr!(L, v)
    emm = L * mu
    varemm = L * varmu * L'
    se = sqrt.(diag(varemm))
    datemm = hcat(V, DataFrame(pred = emm, se = se))
    datemm
    (datemm = datemm, datmu, varemm)
end 
