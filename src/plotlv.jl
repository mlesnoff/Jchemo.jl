"""
    plotlv(T; size = (700, 350), shape, start = 1, color = nothing, zeros::Bool = false,
        xlabel = "", ylabel = "", title = "", kwargs...)
    plotlv(T, group; size = (700, 350), shape, start = 1, color = nothing, zeros::Bool = false,
        xlabel = "", ylabel = "", title = "", leg::Bool = true, leg_title = "Group", 
        kwargs...)
Matrix of plots of successive (PCA, PLS, etc.) latent variables.
* `T` : A matrix of (PCA, PLS, ec.) latent variables (LVs) to plot (n, A).
* `group` : Categorical variable defining groups (n). 
Keyword arguments:
* `size` : Size (horizontal, vertical) of the figure.
* `shape` : A tuple of length = 2 defining the shape of the figure: nb. rows and columns of 
    the matriice of plots. 
* `start` : Start of the numbering of the LVs in the plots.
* `color` : Set color(s). If `group` if used, `color` must be 
    a vector of same length as the number of levels in `group`.
*  `zeros` : Boolean. Draw horizontal and vertical axes passing 
    through origin (0, 0).
*  `xlabel` : Label for the x-axis.
*  `ylabel` : Label for the y-axis.
*  `zlabel` : Label for the z-axis.
*  `title` : Title of the graphic.
* `leg` : Boolean. If `group` is used, display a legend or not.
* `leg_title` : Title of the legend.
* `kwargs` : Optional arguments to pass in function `scatter` of Makie.

To use `plotlv`, a backend (e.g. CairoMakie) has to be specified.

## Examples
```julia

```
""" 
function plotlv(T; size = (700, 350), shape, start = 1, color = nothing, zeros::Bool = false,
        xlabel = "", ylabel = "", title = "", kwargs...)
    n, p = shape
    pmax = nco(T)
    f = Figure(; size)
    isnothing(color) ? color = (:blue, .3) : nothing
    lw = 1.5
    k = 1
    l = copy(start)
    ax = list(Int(n * p))
    for i = 1:n
        for j = 1:p
            if k < pmax
                ax[k] = Axis(f; xlabel = string(xlabel, l), ylabel = string(ylabel, l + 1), title = title)
                scatter!(ax[k], T[:, k], T[:, k + 1]; color = color, kwargs...) 
                if zeros
                    hlines!(0; color = :grey60, linewidth = lw)
                    vlines!(0; color = :grey60, linewidth = lw)
                end
            else
                ax[k] = Axis(f)
                hidespines!(ax[k])
                hidedecorations!(ax[k])
            end
            f[i, j] = ax[k]
            k += 1
            l += 1        
        end
    end
    (f = f, ax)
end

function plotlv(T, group; size = (700, 350), shape, start = 1, color = nothing, zeros::Bool = false,
        xlabel = "", ylabel = "", title = "", leg::Bool = true, leg_title = "Group", 
        kwargs...)
    group = vec(group)
    lev = mlev(group)
    nlev = length(lev)
    lab = string.(lev)
    n, p = shape
    pmax = nco(T)
    ##
    f = Figure(; size)
    lw = 1.5
    posleg = max(round(Int, n / 2), 1)
    k = 1
    l = copy(start)
    ax = list(Int(n * p))
    for i = 1:n
        for j = 1:p
            if k < pmax
                ax[k] = Axis(f; xlabel = string(xlabel, l), ylabel = string(ylabel, l + 1), title = title)
                if isnothing(color)
                    @inbounds for r in eachindex(lev)
                        s = group .== lev[r]                    
                        scatter!(ax[k], T[s, k], T[s, k + 1]; label = lab[r], kwargs...)
                    end
                    if leg && k == 1
                        f[posleg, 0] = Legend(f, ax[k], leg_title; nbanks = 1, rowgap = 10, framevisible = false)
                    end
                else
                    @inbounds for r in eachindex(lev)
                        s = group .== lev[r]                    
                        scatter!(ax[k], T[s, k], T[s, k + 1]; label = lab[r], color = color[r], kwargs...)
                    end
                    if leg && k == 1
                        elt = [MarkerElement(color = color[i], marker = '●', markersize = 10) for i in 1:nlev]
                        f[posleg, 0] = Legend(f, elt, lev, leg_title; nbanks = 1, rowgap = 10, framevisible = false) 
                    end
                end
                if zeros
                    hlines!(0; color = :grey60, linewidth = lw)
                    vlines!(0; color = :grey60, linewidth = lw)
                end
            else
                ax[k] = Axis(f)
                hidespines!(ax[k])
                hidedecorations!(ax[k])
            end
            f[i, j] = ax[k]
            k += 1
            l += 1      
        end
    end
    (f = f, ax, lev)
end

