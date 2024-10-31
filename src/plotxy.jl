"""
    plotxy(x, y; size = (500, 300), color = nothing, ellipse::Bool = false, 
        prob = .95, circle::Bool = false, bisect::Bool = false, zeros::Bool = false,
        xlabel = "", ylabel = "", title = "", kwargs...)
    plotxy(x, y, group; size = (600, 350), color = nothing, ellipse::Bool = false, 
        prob = .95, circle::Bool = false, bisect::Bool = false, zeros::Bool = false,
        xlabel = "", ylabel = "", title = "", leg::Bool = true, title_leg = "Group", 
        kwargs...)
Scatter plot of (x, y) data
* `x` : A x-vector (n).
* `y` : A y-vector (n). 
* `group` : Categorical variable defining groups (n). 
Keyword arguments:
* `size` : Size (horizontal, vertical) of the figure.
* `color` : Set color(s). If `group` if used, `color` must be 
    a vector of same length as the number of levels in `group`.
* `ellipse` : Boolean. Draw an ellipse of confidence, 
    assuming a Ch-square distribution with df = 2. If `group` 
    is used, one ellipse is drawn per group.
* `prob` : Probability for the ellipse of confidence.
*  `bisect` : Boolean. Draw a bisector.
*  `zeros` : Boolean. Draw horizontal and vertical axes passing 
    through origin (0, 0).
*  `xlabel` : Label for the x-axis.
*  `ylabel` : Label for the y-axis.
*  `title` : Title of the graphic.
* `leg` : Boolean. If `group` is used, display a legend 
    or not.
* `title_leg` : Title of the legend.
* `kwargs` : Optional arguments to pass in function `scatter` 
    of Makie.

To use `plotxy`, a backend (e.g. CairoMakie) has 
to be specified.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
lev = mlev(year)
nlev = length(lev)

model = pcasvd(nlv = 5)  
fit!(model, X) 
@head T = model.fitm.T

plotxy(T[:, 1], T[:, 2]; color = (:red, .5)).f

plotxy(T[:, 1], T[:, 2], year; ellipse = true, xlabel = "PC1", ylabel = "PC2").f

i = 2
colm = cgrad(:Dark2_5, nlev; categorical = true)
plotxy(T[:, i], T[:, i + 1], year; color = colm, xlabel = string("PC", i), 
    ylabel = string("PC", i + 1), zeros = true, ellipse = true).f

plotxy(T[:, 1], T[:, 2], year).lev

plotxy(1:5, 1:5).f

y = reshape(rand(5), 5, 1)
plotxy(1:5, y).f

## Several layers can be added
## (same syntax as in Makie)
A = rand(50, 2)
f, ax = plotxy(A[:, 1], A[:, 2]; xlabel = "x1", ylabel = "x2")
ylims!(ax, -1, 2)
hlines!(ax, 0.5; color = :red, linestyle = :dot)
f
```
""" 
function plotxy(x, y; size = (500, 300), color = nothing, ellipse::Bool = false, 
        prob = .95, circle::Bool = false, bisect::Bool = false, zeros::Bool = false,
        xlabel = "", ylabel = "", title = "", kwargs...)
    x = vec(x)
    y = vec(y)
    f = Figure(size = size)
    ax = Axis(f; xlabel = xlabel, ylabel = ylabel, title = title)
    lw = .8
    if isnothing(color)
        scatter!(ax, x, y; kwargs...)
    else
        scatter!(ax, x, y; color = color, kwargs...)
    end
    if ellipse
        X = hcat(x, y)
        xmeans = colmean(X)
        radius = sqrt(quantile(Distributions.Chi(2), prob))
        res = Jchemo.ellipse(cov(X); mu = xmeans, radius)
        if isnothing(color)
            lines!(ax, res.X; color = :grey40, linewidth = lw)
        else
            lines!(ax, res.X; color = color, linewidth = lw)
        end 
    end
    if circle
        z = Jchemo.ellipse(diagm(ones(2))).X
        lines!(ax, z; color = :grey60, linewidth = lw)
    end
    if bisect
        ablines!(ax, 0, 1; color = :grey, linewidth = lw)
    end
    if zeros
        hlines!(0; color = :grey60, linewidth = lw)
        vlines!(0; color = :grey60, linewidth = lw)
    end
    f[1, 1] = ax
    (f = f, ax = ax)
end

function plotxy(x, y, group; size = (600, 350), color = nothing, ellipse::Bool = false, 
        prob = .95, circle::Bool = false, bisect::Bool = false, zeros::Bool = false,
        xlabel = "", ylabel = "", title = "", leg::Bool = true, title_leg = "Group", 
        kwargs...)
    x = vec(x)
    y = vec(y)
    group = vec(group)
    lev = mlev(group)
    lab = string.(lev)
    f = Figure(size = size)
    ax = Axis(f; xlabel = xlabel, ylabel = ylabel, title = title)
    lw = .8
    @inbounds for i in eachindex(lev)
        s = group .== lev[i]
        zx = x[s]
        zy = y[s]
        if isnothing(color)
            scatter!(ax, zx, zy; label = lab[i], kwargs...)
        else
            scatter!(ax, zx, zy; label = lab[i], color = color[i], kwargs...)
        end
        if ellipse
            X = hcat(zx, zy)
            xmeans = colmean(X)
            radius = sqrt(quantile(Chi(2), prob))
            res = Jchemo.ellipse(cov(X); mu = xmeans, radius)
            if isnothing(color)
                lines!(ax, res.X; color = :grey40, linewidth = lw)
            else
                lines!(ax, res.X; color = color[i], linewidth = lw)
            end 
        end
    end
    if circle
        res = Jchemo.ellipse(diagm(ones(2))).X
        lines!(ax, res; color = :grey60, linewidth = lw)
    end
    if bisect
        ablines!(ax, 0, 1; linewidth = lw)
    end
    if zeros
        hlines!(0; color = :grey60, linewidth = lw)
        vlines!(0; color = :grey60, linewidth = lw)
    end
    f[1, 1] = ax
    if leg
        f[1, 2] = Legend(f, ax, title_leg, framevisible = false)
    end
    (f = f, ax = ax, lev = lev)
end

