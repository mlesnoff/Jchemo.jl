"""
    plotscore(x, y; resolution = (500, 350), step = 5, 
        color = nothing, kwargs...)
    plotscore(x, y, group; resolution = (500, 350), step = 5, 
        color = nothing, kwargs...)

Plotting model performances (scores) after validation.
* `x` : A variable representing the parameter(s).
* `y` : The model scores (e.g. error rates) for the values of `x`. 
* `group` : Grouping parameters if multiple.
* 'resolution' : Resolution (horizontal, vertical) of the figure.
* `color` : Set color. If `group` if used, must be a vector of same length
    as the number of levels in `group`.
* `kwargs` : Optional arguments to pass in `Axis`.

## Examples
```julia
using JLD2, CairoMakie
mypath = dirname(dirname(pathof(Jchemo)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
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
plotscore(res.nlv, res.y1;
    xlabel = "Nb. LVs", ylabel = "RMSEP").f

nlvdis = 15 ; metric = ["mahal" ;]
h = [1 ; 2.5 ; 5] ; k = [50 ; 100] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k)
nlv = 0:20
res = gridscorelv(Xtrain, ytrain, Xtest, ytest;
    score = rmsep, fun = lwplsr, pars = pars, nlv = nlv)
group = string.("h=", res.h, " k=", res.k)
plotscore(res.nlv, res.y1, group;
    xlabel = "Nb. LVs", ylabel = "RMSECV").f
```
""" 
function plotscore(x, y; resolution = (500, 350), step = 5, 
        color = nothing, kwargs...)
    x = vec(x)
    y = vec(y)
    xticks = collect(minimum(x):step:maximum(x))
    f = Figure(resolution = resolution)
    ax = Axis(f; xticks = (xticks, string.(xticks)), kwargs...)
    if isnothing(color)
        lines!(ax, x, y)
    else
        lines!(ax, x, y; color = color)
    end
    f[1, 1] = ax
    (f = f, ax = ax)
end

function plotscore(x, y, group; resolution = (700, 350), step = 5, 
        color = nothing, kwargs...)
    x = vec(x)
    y = vec(y)
    group = vec(group)
    xticks = collect(minimum(x):step:maximum(x))
    lev = sort(unique(group))
    nlev = length(lev)
    println(nlev)
    f = Figure(resolution = resolution)
    ax = Axis(f; xticks = (xticks, string.(xticks)), kwargs...)
    for i = 1:nlev
        s = group .== lev[i]
        zx = x[s]
        zy = y[s]
        lab = string(lev[i])
        if isnothing(color)
            lines!(ax, zx, zy; label = lab)
        else
            lines!(ax, zx, zy; label = lab, color = color[i])
        end
    end
    f[1, 1] = ax
    f[1, 2] = Legend(f, ax, "Group", framevisible = false)
    (f = f, ax = ax)
end

