"""
    plotgrid(indx::AbstractVector, r; size = (500, 300), step = 5, 
        color = nothing, kwargs...)
    plotgrid(indx::AbstractVector, r, group; size = (700, 350), 
        step = 5, color = nothing, leg = true, leg_title = "Group", kwargs...)
Plot error/performance rates of a model.
* `indx` : A numeric variable representing the grid of 
    model parameters, e.g. the nb. LVs if PLSR models.
* `r` : The error/performance rate.
Keyword arguments: 
* `group` : Categorical variable defining groups. 
    A separate line is plotted for each level of `group`.
* `size` : Size (horizontal, vertical) of the figure.
* `step` : Step used for defining the xticks.
* `color` : Set color. If `group` if used, must be a vector 
    of same length as the number of levels in `group`.
* `leg` : Boolean. If `group` is used, display a legend or not.
* `leg_title` : Title of the legend.
* `kwargs` : Optional arguments to pass in `Axis` of CairoMakie.

To use the function, a backend (e.g. CairoMakie) has to be specified.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

model = plskern() 
nlv = 0:20
res = gridscore(model, Xtrain, ytrain, 
    Xtest, ytest; score = rmsep, nlv)
plotgrid(res.nlv, res.y1; xlabel = "Nb. LVs", ylabel = "RMSEP").f

model = lwplsr() 
nlvdis = 15 ; metric = [:mah]
h = [1 ; 2.5 ; 5] ; k = [50 ; 100] 
pars = mpar(nlvdis = nlvdis, metric = metric, 
    h = h, k = k)
nlv = 0:20
res = gridscore(model, Xtrain, ytrain, Xtest, ytest; score = rmsep, 
    pars, nlv)
group = string.("h=", res.h, " k=", res.k)
plotgrid(res.nlv, res.y1, group; xlabel = "Nb. LVs", ylabel = "RMSECV").f
```
""" 
function plotgrid(indx::AbstractVector, r; size = (500, 300), step = 5, 
        color = nothing, kwargs...)
    isa(indx, Vector{Any}) ? indx = Float64.(indx) : nothing
    r = Float64.(vec(r))
    xticks = collect(minimum(indx):step:maximum(indx))
    f = Figure(; size)
    ax = Axis(f; xticks = (xticks, string.(xticks)), kwargs...)
    if isnothing(color)
        lines!(ax, indx, r)
    else
        lines!(ax, indx, r; color = color)
    end
    f[1, 1] = ax
    (f = f, ax)
end

function plotgrid(indx::AbstractVector, r, group; size = (700, 350), 
        step = 5, color = nothing, leg = true, leg_title = "Group", kwargs...)
    isa(indx, Vector{Any}) ? indx = Float64.(indx) : nothing
    r = Float64.(vec(r))
    group = vec(group)
    xticks = collect(minimum(indx):step:maximum(indx))
    lev = mlev(group)
    f = Figure(; size)
    ax = Axis(f; xticks = (xticks, string.(xticks)), kwargs...)
    @inbounds for i in eachindex(lev)
        s = group .== lev[i]
        x = indx[s]
        y = r[s]
        lab = string(lev[i])
        if isnothing(color)
            lines!(ax, x, y; label = lab)
        else
            lines!(ax, x, y; label = lab, color = color[i])
        end
    end
    f[1, 1] = ax
    if leg
        f[1, 2] = Legend(f, ax, leg_title, framevisible = false)
    end
    (f = f, ax)
end

