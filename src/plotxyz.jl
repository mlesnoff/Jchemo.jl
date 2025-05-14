"""
    plotxy(x, y, z; size = (500, 300), color = nothing, perspectiveness = .1,
        xlabel = "", ylabel = "", zlabel = "", title = "", kwargs...)
    plotxy(x, y, z, group; size = (500, 300), color = nothing, perspectiveness = .1, 
        xlabel = "", ylabel = "", zlabel = "", title = "", leg::Bool = true, leg_title = "Group", 
        kwargs...)
3D scatter plot of x-y-z data.
* `x` : A x-vector (n).
* `y` : A y-vector (n). 
* `z` : A y-vector (n). 
* `group` : Categorical variable defining groups (n). 
Keyword arguments:
* `size` : Size (horizontal, vertical) of the figure.
* `color` : Set color(s). If `group` if used, `color` must be 
    a vector of same length as the number of levels in `group`.
*  `xlabel` : Label for the x-axis.
*  `ylabel` : Label for the y-axis.
*  `zlabel` : Label for the z-axis.
*  `title` : Title of the graphic.
* `leg` : Boolean. If `group` is used, display a legend or not.
* `leg_title` : Title of the legend.
* `kwargs` : Optional arguments to pass in function `scatter` of Makie.

To use `plotxy`, a backend (e.g. CairoMakie) has to be specified.

## Examples
```julia
using Jchemo, CairoMakie, GLMakie
n = 1000
x = randn(n)
y = randn(n)
z = randn(n)
group = rand(["A", "B", "C"], n)
s = group .== "B"
x[s] .+= 10 ;
s = group .== "C"
x[s] .+= 20 ;

CairoMakie.activate!()
#GLMakie.activate!()

plotxyz(x, y, z; size = (500, 300), markersize = 10, xlabel = "V1").f
plotxyz(x, y, z; size = (500, 300), color = (:red, .3), markersize = 10, xlabel = "V1").f

plotxyz(x, y, z, group; size = (500, 300), markersize = 10, xlabel = "V1").f
plotxyz(x, y, z, group; size = (500, 300), markersize = 10, xlabel = "V1", alpha = .3).f

color = [(:red, .3); (:blue, .3); (:green, .3)]
#color = cgrad(:Dark2_5; categorical = true, alpha = .3)[1:nlev]
plotxyz(x, y, z, group; size = (500, 300), color = color, leg = true, markersize = 10, xlabel = "V1").f
```
""" 
function plotxyz(x, y, z; size = (500, 300), color = nothing, perspectiveness = .1,
        xlabel = "", ylabel = "", zlabel = "", title = "", kwargs...)
    x = vec(x)
    y = vec(y)
    z = vec(z)
    f = Figure(; size)
    ax = Axis3(f[1, 1]; xlabel = xlabel, ylabel = ylabel, zlabel = zlabel, title = title, 
        perspectiveness = perspectiveness) 
    isnothing(color) ? color = (:blue, .3) : nothing
    scatter!(ax, x, y, z; color = color, kwargs...)
    f[1, 1] = ax
    (f = f, ax)
end

function plotxyz(x, y, z, group; size = (500, 300), color = nothing, perspectiveness = .1, 
        xlabel = "", ylabel = "", zlabel = "", title = "", leg::Bool = true, leg_title = "Group", 
        kwargs...)
    x = vec(x)
    y = vec(y)
    z = vec(z)
    group = vec(group)
    lev = mlev(group)
    nlev = length(lev)
    lab = string.(lev)    
    f = Figure(; size)
    ax = Axis3(f[1, 1]; xlabel = xlabel, ylabel = ylabel, zlabel = zlabel, title = title, 
        perspectiveness = perspectiveness)
    if isnothing(color)
        @inbounds for i in eachindex(lev)
            s = group .== lev[i]
            scatter!(ax, x[s], y[s], z[s]; label = lab[i], kwargs...)
        end
        if leg
            axislegend(ax, leg_title; position = :rc, nbanks = 1, rowgap = 10, framevisible = false)
            #f[1, 2] = Legend(f, ax, leg_title; nbanks = 1, rowgap = 10, framevisible = false) 
        end
    else
        @inbounds for i in eachindex(lev)
            s = group .== lev[i]
            scatter!(ax, x[s], y[s], z[s]; label = lab[i], color = color[i], kwargs...)
        end
        if leg
            elt = [MarkerElement(color = color[i], marker = '●', markersize = 10) for i in 1:nlev]
            Legend(f[1, 2], elt, lev, leg_title; nbanks = 1, rowgap = 10, framevisible = false)
        end
    end
    (f = f, ax = ax, lev = lev)
end

