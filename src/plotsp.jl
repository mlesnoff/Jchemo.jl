"""
    plotsp(X, wl = 1:nco(X); size = (500, 300),
        color = nothing, nsamp = nothing, kwargs...)
Plotting spectra.
* `X` : X-data (n, p).
* `wl` : Column names of `X`. Must be numeric.
Keyword arguments:
* `size` : Size (horizontal, vertical) of the figure.
* `color` : Set a unique color (and eventually transparency) 
    to the spectra.
* `nsamp` : Nb. spectra (X-rows) to plot. If `nothing`, 
    all spectra are plotted.
* `kwargs` : Optional arguments to pass in `Axis` of CairoMakie.

The function plots the rows of `X`.

To use `plotxy`, a backend (e.g. CairoMakie) has 
to be specified.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X
wlst = names(X)
wl = parse.(Float64, wlst) 

plotsp(X).f
plotsp(X; color = (:red, .2)).f
plotsp(X, wl; xlabel = "Wavelength (nm)", ylabel = "Absorbance").f

f, ax = plotsp(X, wl; color = (:red, .2))
xmeans = colmean(X)
lines!(ax, wl, xmeans; color = :black, linewidth = 2)
vlines!(ax, 1200)
f
```

""" 
function plotsp(X, wl = 1:nco(X); size = (500, 300),
        color = nothing, nsamp = nothing, kwargs...) 
    X = ensure_mat(X)
    if !isnothing(nsamp)
        X = X[sample(1:nro(X), nsamp; replace = false), :]
    end
    ## For not using function Size
    ## (conflict with argiment size of Makie)
    n = nro(X)
    p = nco(X)
    ## End
    f = Figure(size = size)
    ax = Axis(f; kwargs...)
    res = list(Matrix{Float64}, n)
    if isnothing(color)
        s = randperm(n)
        for i = 1:n
            res[i] = hcat([wl ; NaN], [X[i, :] ; NaN], s[i] * ones(p + 1))
        end
        res = reduce(vcat, res)
        tp = .9
        if n == 1
            lines!(ax, res[:, 1], res[:, 2]; color = "red3")
        else
            #colm = (:Paired_12, tp)
            #colm = (:seaborn_bright, tp)
            colm = (:Set1_9, tp)
            lines!(ax, res[:, 1], res[:, 2]; colormap = colm, 
                color = res[:, 3])
        end
    else
        for i = 1:n
            res[i] = hcat([wl ; NaN], [X[i, :] ; NaN])
        end
        ## Same as:
        ## res = [hcat([wl ; NaN], [X[i, :] ; NaN]) for i = 1:n]
        ## End
        res = reduce(vcat, res)
        lines!(ax, res[:, 1], res[:, 2]; color = color)
    end
    f[1, 1] = ax
    (f = f, ax = ax)
end



