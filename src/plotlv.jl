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

