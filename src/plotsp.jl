"""
    plotsp(X, wl = 1:nco(X); size = (500, 300), nsamp = nro(X), color = nothing, 
        kwargs...)
Plotting spectra.
* `X` : X-data (n, p).
* `wl` : Column names of `X`. Must be numeric.
Keyword arguments:
* `size` : Size (horizontal, vertical) of the figure.
* `nsamp` : Nb. spectra (X-rows) to plot. If `nothing`, 
    all spectra are plotted.
* `color` : Set a unique color (and eventually transparency) 
    to the spectra.
* `kwargs` : Optional arguments to pass in `Axis` of CairoMakie.

The function plots the rows of `X`.

To use `plotxy`, a backend (e.g. CairoMakie) has 
to be specified.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
X = dat.X
wlst = names(X)
wl = parse.(Float64, wlst) 

plotsp(X).f
plotsp(X; color = (:red, .2)).f
plotsp(X, wl; xlabel = "Wavelength (nm)", ylabel = "Absorbance").f

tck = collect(wl[1]:200:wl[end]) ;
plotsp(X, wl; xlabel = "Wavelength (nm)", ylabel = "Absorbance", xticks = tck).f

f, ax = plotsp(X, wl; color = (:red, .2))
xmeans = colmean(X)
lines!(ax, wl, xmeans; color = :black, linewidth = 2)
vlines!(ax, 1200)
f
```

""" 
function plotsp(X, wl = 1:nco(X); size = (500, 300), nsamp = nro(X), color = nothing, 
        kwargs...) 
    X = ensure_mat(X)
    n, p = Base.size(X)    # conflict with Makie.size
    s = StatsBase.sample(1:n, nsamp; replace = false)
    vX = vrow(X, s)
    m = nro(vX)
    f = Figure(; size)
    ax = Axis(f; kwargs...)
    res = list(Matrix{Float64}, m)
    if isnothing(color)
        s = randperm(m)
        for i in eachindex(res)
            res[i] = hcat([wl ; NaN], [vX[i, :] ; NaN], s[i] * ones(p + 1))
        end
        res = reduce(vcat, res)
        transp = .9
        if m == 1
            lines!(ax, res[:, 1], res[:, 2]; color = "red3")
        else
            #colm = (:Paired_12, transp)
            #colm = (:seaborn_bright, transp)
            colm = (:Set1_9, transp)
            lines!(ax, res[:, 1], res[:, 2]; colormap = colm, color = res[:, 3])
        end
    else
        for i in eachindex(res)
            res[i] = hcat([wl ; NaN], [vX[i, :] ; NaN])
        end
        ## Same as:
        ## res = [hcat([wl ; NaN], [vX[i, :] ; NaN]) for i in eachindex(res)]
        ## End
        res = reduce(vcat, res)
        lines!(ax, res[:, 1], res[:, 2]; color = color)
    end
    f[1, 1] = ax
    (f = f, ax = ax)
end



