"""
    plotsp(X, wl = 1:size(X, 2); resolution = (500, 350),
        color = nothing, kwargs...) 
Plotting spectra.
* `X` : X-data.
* `wl` : Column names of `X`. Must be numeric.
* `color` : Set a unique color (and eventually transparency) to the spectra.
* 'resolution' : Resolution (horizontal, vertical) of the figure.
* `kwargs` : Optional arguments to pass in `Axis`.
The function plots the rows of `X`.

## Examples
```julia
    using JLD2, CairoMakie
    mypath = dirname(dirname(pathof(Jchemo)))
    db = joinpath(mypath, "data", "cassav.jld2") 
    @load db dat
    pnames(dat)
    
    X = dat.X
    wl = names(X)
    wl_num = parse.(Float64, wl) 
    plotsp(X).f
    plotsp(X; color = (:red, .2)).f
    plotsp(X, wl_num; xlabel = "Wavelength (nm)",
        ylabel = "Absorbance").f

    f, ax = plotsp(X)
    vlines!(ax, 200)
    f
```

""" 
function plotsp(X, wl = 1:size(X, 2); resolution = (500, 350),
        color = nothing, kwargs...) 
    X = ensure_mat(X)
    n, p = size(X)
    f = Figure(resolution = resolution)
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



