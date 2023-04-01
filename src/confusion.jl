
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
    accur = sum(diag(A)) / n 
    (cnt = cnt, pct, A, Apct, accur, lev)
end

function plotconf(object; pct = true, ptext = false, 
        fontsize = 15, resolution = (500, 400))
    if pct
        A = object.Apct 
        namval = "Accuracy (%)"
    else
        A = object.A 
        namval = "Nb. occurence"
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
            i == j ? col = :red : col = :white
            text!(ax, string(val); position = (j, nlev - i + 1), 
                align = (:center, :center), fontsize = fontsize,
                color = col)
        end
    end
    ax.xticklabelrotation = π / 3   # default: 0
    ax.xticklabelalign = (:right, :center)
    (f = f, ax, hm)
end




