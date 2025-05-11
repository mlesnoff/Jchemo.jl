function plotxyz(x, y, z; size = (500, 300), color = nothing, perspectiveness = .1,
        xlabel = "", ylabel = "", zlabel = "", title = "", kwargs...)
    x = vec(x)
    y = vec(y)
    z = vec(z)
    f = Figure(; size)
    ax = Axis3(f[1, 1]; xlabel = xlabel, ylabel = ylabel, zlabel = zlabel, title = title, 
        perspectiveness = perspectiveness) 
    if isnothing(color)
        scatter!(ax, x, y, z; kwargs...)
    else
        scatter!(ax, x, y, z; color = color, kwargs...)
    end
    f[1, 1] = ax
    (f = f, ax = ax)
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
    @inbounds for i in eachindex(lev)
        s = group .== lev[i]
        if isnothing(color)
            scatter!(ax, x[s], y[s], z[s]; label = lab[i], kwargs...)
        else
            scatter!(ax, x[s], y[s], z[s]; label = lab[i], color = color[i], kwargs...)
        end
    end
    if leg
        elt = [MarkerElement(color = color[i], marker = '●', markersize = 10) for i in 1:nlev]
        Legend(f[1, 2], elt, lev, leg_title; nbanks = 1, rowgap = 10, framevisible = false)
    end
    (f = f, ax = ax, lev = lev)
end

