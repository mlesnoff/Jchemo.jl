"""
    plotsp(X, wl = 1:size(X, 2); color = (:blue, .5), kwargs...) 
Plotting spectra (faster than `plotsp`).
* `X` : X-data.
* `wl` : Column names of `X`. Must be numeric.
* `color` : Set a unique color (and eventually transparency) to the spectra.
* `kwargs` : Optional arguments to pass in `Axis`.
Plots lines corresponding to the rows of `X`.
""" 
function plotsp(X, wl = 1:size(X, 2); color = nothing, kwargs...) 
    X = ensure_mat(X)
    n, p = size(X)
    f = Figure()
    ax = Axis(f; kwargs...)
    res = Vector{Matrix}(undef, n)
    if isnothing(color)
        k = randperm(n)
        for i = 1:n
            res[i] = hcat([wl ; NaN], [X[i, :] ; NaN], k[i] * ones(p + 1))
        end
        res = reduce(vcat, res)
        tp = .9
        if n == 1
            lines!(ax, res[:, 1], res[:, 2]; color = "red3")
        else
            #cm = (:Paired_12, tp)
            #cm = (:seaborn_bright, tp)
            cm = (:Set1_9, tp)
            lines!(ax, res[:, 1], res[:, 2]; colormap = cm, color = res[:, 3])
        end
    else
        for i = 1:n
            res[i] = hcat([wl ; NaN], [X[i, :] ; NaN])
        end
        res = reduce(vcat, res)
        lines!(ax, res[:, 1], res[:, 2]; color = color)
    end
    f[1, 1] = ax
    (f = f, ax = ax)
end



