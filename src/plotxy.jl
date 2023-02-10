"""
    plotxy(x, y; resolution = (500, 400), 
        color = nothing, ellipse::Bool = false, prob = .95, 
        circle::Bool = false, bisect::Bool = false, zeros::Bool = false, 
        kwargs...)
    plotxy(x, y, group; resolution = (600, 400), 
        color = nothing, ellipse::Bool = false, prob = .95, 
        circle::Bool = false, bisect::Bool = false, zeros::Bool = false, 
        kwargs...)
        
Scatter plot of (x, y) data
* `x` : A x-variable (n).
* `y` : A y-variable (n). 
* `group` : Categorical variable defining groups. 
    A separate line is plotted for each level of `group`.
* 'resolution' : Resolution (horizontal, vertical) of the figure.
* `color` : Set color. If `group` if used, must be a vector of same length
    as the number of levels in `group`.
* `ellipse` : Boolean. Draw an ellipse of confidence, assuming a Ch-square distribution
    with df = 2. If `group` is used, one ellipse per group is drawn.
* `prob` : Probability for the ellipse of confidence (default = .95).
*  `bisect` : Boolean. Draw a bisector.
*  `zeros` : Boolean. Draw horizontal and vertical axes passing through origin (0, 0).
* `kwargs` : Optional arguments to pass in `Axis` of CairoMakie.

The user has to specify a backend (e.g. CairoMakie).

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)

fm = pcasvd(X, nlv = 3) ; 
T = fm.T

plotxy(T[:, 1], T[:, 2]; color = (:red, .5)).f
plotxy(T[:, 1], T[:, 2], year; ellipse = true).f
```
""" 
function plotxy(x, y; resolution = (500, 400), 
        color = nothing, ellipse::Bool = false, prob = .95, 
        circle::Bool = false, bisect::Bool = false, zeros::Bool = false, 
        kwargs...)
    f = Figure(resolution = resolution)
    ax = Axis(f; kwargs...)
    if isnothing(color)
        scatter!(ax, x, y)
    else
        scatter!(ax, x, y; color = color)
    end
    if ellipse
        X = hcat(x, y)
        xmeans = colmean(X)
        radius = sqrt(quantile(Chi(2), prob))
        res = Jchemo.ellipse(cov(X); center = xmeans, radius = radius)
        if isnothing(color)
            lines!(ax, res.X; color = :grey40)
        else
            lines!(ax, res.X; color = color)
        end 
    end
    if circle
        z = Jchemo.ellipse(diagm(ones(2))).X
        lines!(ax, z; color = :grey60)
    end
    if bisect
        ablines!(ax, 0, 1)
    end
    if zeros
        hlines!(0; color = :grey60)
        vlines!(0; color = :grey60)
    end
    f[1, 1] = ax
    (f = f, ax = ax)
end

function plotxy(x, y, group; resolution = (600, 400), 
        color = nothing, ellipse::Bool = false, prob = .95, 
        circle::Bool = false, bisect::Bool = false, zeros::Bool = false, 
        kwargs...)
    group = vec(group)
    lev = sort(unique(group))
    nlev = length(lev)
    lab = string.(lev)
    f = Figure(resolution = resolution)
    ax = Axis(f; kwargs...)
    for i = 1:nlev
        s = group .== lev[i]
        zx = x[s]
        zy = y[s]
        if isnothing(color)
            scatter!(ax, zx, zy; label = lab[i])
        else
            scatter!(ax, zx, zy; label = lab[i], color = color[i])
        end
        if ellipse
            X = hcat(zx, zy)
            xmeans = colmean(X)
            radius = sqrt(quantile(Chi(2), prob))
            res = Jchemo.ellipse(cov(X); center = xmeans, radius = radius)
            if isnothing(color)
                lines!(ax, res.X; color = :grey40)
            else
                lines!(ax, res.X; color = color[i])
            end 
        end
    end
    if circle
        res = Jchemo.ellipse(diagm(ones(2))).X
        lines!(ax, res; color = :grey60)
    end
    if bisect
        ablines!(ax, 0, 1)
    end
    if zeros
        hlines!(0; color = :grey60)
        vlines!(0; color = :grey60)
    end
    f[1, 1] = ax
    f[1, 2] = Legend(f, ax, "Group", framevisible = false)
    (f = f, ax = ax)
end

