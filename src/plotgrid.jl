"""
    plotgrid(indx::Union{Vector{Integer}, Vector{Int64}, Vector{Real}, Vector{Float64}}, r; 
        resolution = (500, 350), step = 5, 
        color = nothing, kwargs...)
    plotgrid(indx::Union{Vector{Integer}, Vector{Int64}, Vector{Real}, Vector{Float64}}, r, 
        group; 
        resolution = (500, 350), step = 5, 
        color = nothing, kwargs...)

Plot error or performance rates of model predictions.
* `indx` : A numeric variable representing the grid of model parameters, 
    e.g. nb. LVs if PLSR models.
* `r` : The error/performance rates for the values of `x`. 
* `group` : Categorical variable defining groups. 
    A separate line is plotted for each level of `group`.
* `resolution` : Resolution (horizontal, vertical) of the figure.
* `step` : Step used for defining the xticks.
* `color` : Set color. If `group` if used, must be a vector of same length
    as the number of levels in `group`.
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
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 0:20
res = gridscorelv(Xtrain, ytrain, Xtest, ytest;
    score = rmsep, fun = plskern, nlv = nlv)
plotgrid(res.nlv, res.y1;
    xlabel = "Nb. LVs", ylabel = "RMSEP").f

nlvdis = 15 ; metric = ["mahal" ]
h = [1 ; 2.5 ; 5] ; k = [50 ; 100] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k)
nlv = 0:20
res = gridscorelv(Xtrain, ytrain, Xtest, ytest;
    score = rmsep, fun = lwplsr, pars = pars, nlv = nlv)
group = string.("h=", res.h, " k=", res.k)
plotgrid(res.nlv, res.y1, group;
    xlabel = "Nb. LVs", ylabel = "RMSECV").f
```
""" 
function plotgrid(indx::Union{Vector{Integer}, Vector{Int64}, Vector{Real}, Vector{Float64}}, r; 
        resolution = (500, 350), step = 5, 
        color = nothing, kwargs...)
    r = Float64.(vec(r))
    xticks = collect(minimum(indx):step:maximum(indx))
    f = Figure(resolution = resolution)
    ax = Axis(f; xticks = (xticks, string.(xticks)), kwargs...)
    if isnothing(color)
        lines!(ax, indx, r)
    else
        lines!(ax, indx, r; color = color)
    end
    f[1, 1] = ax
    (f = f, ax = ax)
end

function plotgrid(indx::Union{Vector{Integer}, Vector{Int64}, Vector{Real}, Vector{Float64}}, r, 
        group; 
        resolution = (700, 350), step = 5, 
        color = nothing, kwargs...)
    r = Float64.(vec(r))
    group = vec(group)
    xticks = collect(minimum(indx):step:maximum(indx))
    lev = sort(unique(group))
    nlev = length(lev)
    f = Figure(resolution = resolution)
    ax = Axis(f; xticks = (xticks, string.(xticks)), kwargs...)
    for i = 1:nlev
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
    f[1, 2] = Legend(f, ax, "Group", framevisible = false)
    (f = f, ax = ax)
end

