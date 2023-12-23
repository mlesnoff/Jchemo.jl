"""
    plotconf(object; size = (500, 400), 
        cnt = true, ptext = true, fontsize = 15, 
        coldiag = :red, )
Plot a confusion matrix.
* `object` : Output of function `confusion`.
Keyword arguments:
* `size` : Size (horizontal, vertical) of the figure.
* `cnt` : Boolean. If `true`, plot the occurrences, 
    else plot the row %s.
* `ptext` : Boolean. If `true`, display the value in each cell.
* `fontsize` : Font size when `ptext = true`.
* `coldiag` : Font color when `ptext = true`.

See examples in help page of function `confusion`.
```
"""
function plotconf(object; size = (500, 400), 
        cnt = true, ptext = true, fontsize = 15, 
        coldiag = :red, )
    if cnt
        A = object.A 
        namval = "Nb. occurrences"
    else
        A = object.Apct 
        namval = "Row %"
    end
    zA = (A')[:, end:-1:1]
    lev = string.(object.lev)
    nlev = length(lev)
    f = Figure(size = size)
    ax = Axis(f[1, 1], xlabel = "Predicted", ylabel = "Observed", 
        xticks = (1:nlev, lev), yticks = (1:nlev, lev[end:-1:1]))
    hm = heatmap!(ax, 1:nlev, 1:nlev, zA)
    Colorbar(f[:, end + 1], hm; label = namval)
    if ptext
        for i = 1:nlev, j = 1:nlev
            cnt ? val = A[i, j] : val = round(A[i, j]; digits = 1)
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




