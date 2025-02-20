"""
    conf(pred, y; digits = 1)
Confusion matrix.
* `pred` : Univariate predictions.
* `y` : Univariate observed data.
Keyword arguments:
* `digits` : Nb. digits used to round percentages.

## Examples
```julia
using Jchemo, CairoMakie

y = ["d"; "c"; "b"; "c"; "a"; "d"; "b"; "d"; 
    "b"; "b"; "a"; "a"; "c"; "d"; "d"]
pred = ["a"; "d"; "b"; "d"; "b"; "d"; "b"; "d"; 
    "b"; "b"; "a"; "a"; "d"; "d"; "d"]
#y = rand(1:10, 200); pred = rand(1:10, 200)

res = conf(pred, y) ;
@names res
res.cnt       # Counts (dataframe built from `A`) 
res.pct       # Row %  (dataframe built from `Apct`))
res.A         
res.Apct
res.diagpct
res.accpct    # Accuracy (% classification successes)
res.lev       # Levels

plotconf(res).f

plotconf(res; cnt = false, ptext = false).f
```
"""
function conf(pred, y; digits = 1)
    pred = vec(pred)
    y = vec(y)
    n = length(y)
    z = vcat(y, pred)
    lev = mlev(z) 
    nlev = length(lev)
    namy = string.(lev)
    nampred = string.("pred_", lev)
    A = Int.(zeros(nlev, nlev))
    @inbounds for i in eachindex(lev)
        @inbounds for j in eachindex(lev)
            zy = y .== lev[i]
            zpred = pred .== lev[j]
            A[i, j] = sum(zy .* zpred)
        end
    end
    rowtot = rowsum(A)
    Apct = 100 * A ./ rowtot
    cnt = DataFrame(A, nampred)
    insertcols!(cnt, 1, :y => namy)
    pct = DataFrame(round.(Apct; digits = digits), nampred)
    insertcols!(pct, 1, :levels => namy)
    diagpct = DataFrame(lev = lev, errp_pct = round.(100 .- diag(Apct) ; digits))
    accpct = round(100 * sum(diag(A)) / n; digits)
    maccpct = round(mean(diag(Apct))[1]; digits)
    accpct = DataFrame(typ = ["Overall", "Mean by class"], accuracy_pct = [accpct ; maccpct])
    (cnt = cnt, pct, A, Apct, diagpct, accpct, lev)
end


