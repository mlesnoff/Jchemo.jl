"""
    confusion(pred, y)
Confusion matrix.
* `pred` : Univariate predictions.
* `y` : Univariate observed data.
* `digits` : Nb. digits used to round percentages.

## Examples
```julia
y = ["d"; "c"; "b"; "c"; "a"; "d"; "b"; "d"; 
    "b"; "b"; "a"; "a"; "c"; "d"; "d"]
pred = ["a"; "d"; "b"; "d"; "b"; "d"; "b"; "d"; 
    "b"; "b"; "a"; "a"; "d"; "d"; "d"]
#y = rand(1:10, 200); pred = rand(1:10, 200)

res = confusion(pred, y) ;
pnames(res)
res.cnt       # Counts (dataframe built from `A`) 
res.pct       # Row %  (dataframe built from `Apct`))
res.A         
res.Apct
res.accuracy  # Overall accuracy (% classification successes)
res.lev       # Levels

plotconf(res).f

plotconf(res; pct = true, ptext = false).f
```
"""
function confusion(pred, y; digits = 1)
    y = vec(y)
    pred = vec(pred)
    n = length(y)
    z = vcat(y, pred)
    lev = mlev(z) 
    nlev = length(lev)
    namy = string.(lev)
    nampred = string.("pred_", lev)
    A = Int64.(zeros(nlev, nlev))
    for i = 1:nlev
        for j = 1:nlev
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
    accuracy = sum(diag(A)) / n 
    (cnt = cnt, pct, A, Apct, accuracy, lev)
end

"""
    plotconf(object; pct = false, ptext = true, 
        fontsize = 15, coldiag = :red, resolution = (500, 400))
Plot a confusion matrix.
* `object` : Output of function `confusion`.
* `pct` : Boolean. If `true` (default), plot the occurrences, 
    else plot the row %s.
* `ptext` : Boolean. If `true` (default), display the value in each cell.
* `fontsize` : Font size when `ptext = true`.
* `coldiag` : Font color when `ptext = true`.
* `resolution` : Resolution (horizontal, vertical) of the figure.

See examples in help page of function `confusion`.
```
"""
function plotconf(object; pct = false, ptext = true, 
        fontsize = 15, coldiag = :red, resolution = (500, 400))
    if pct
        A = object.Apct 
        namval = "Row %"
    else
        A = object.A 
        namval = "Nb. occurences"
    end
    zA = (A')[:, end:-1:1]
    lev = string.(object.lev)
    nlev = length(lev)
    f = Figure(resolution = resolution)
    ax = Axis(f[1, 1], xlabel = "Predicted", ylabel = "Observed", 
        xticks = (1:nlev, lev), yticks = (1:nlev, lev[end:-1:1]))
    hm = heatmap!(ax, 1:nlev, 1:nlev, zA)
    Colorbar(f[:, end + 1], hm; label = namval)
    if ptext
        for i = 1:nlev, j = 1:nlev
            pct ? val = round(A[i, j]; digits = 1) : val = A[i, j]
            i == j ? col = coldiag : col = :white
            text!(ax, string(val); position = (j, nlev - i + 1), 
                align = (:center, :center), fontsize = fontsize,
                color = col)
        end
    end
    ax.xticklabelrotation = π / 3   # default: 0
    ax.xticklabelalign = (:right, :center)
    (f = f, ax, hm)
end




