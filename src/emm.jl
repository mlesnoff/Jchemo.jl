"""
    emm())
Estimated marginal means.
* `` :.
Keyword arguments:
* `` :. 

## References

## Examples 
```julia
```
"""
function emm(fitm::StatsModels.TableRegressionModel, f::StatsModels.FormulaTerm, dat::DataFrame)

    f_fitm = StatsModels.formula(fitm)    # = fitm.mf.f    
    b = StatsAPI.coef(fitm)
    varb = StatsAPI.vcov(fitm)   # to do: check if correct for glims

    ## Build the full table corresponding to model fitm
    nam = string.(@names dat)
    u = in(StatsModels.termnames(f_fitm.rhs)).(nam)
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
    #tmp = copy(datmu)
    #tmp.y = zeros(nro(tmp))  # to remove: instead, write directely a rhs formula
    #mf = ModelFrame(f_fitm, tmp)
    #fs = apply_schema(f_fitm, mf.schema)
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
    res = (datemm = datemm, datmu, varemm) ;
end 
