function plotlv(T; size = (700, 350), shape, start = 1, 
        color = nothing, 
        #ellipse::Bool = false, prob = .95, circle::Bool = false
        zeros::Bool = false,
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
                ax[k] = Axis(f; xlabel = string(xlabel, l), ylabel = string(ylabel, l + 1))
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
    (f = f, ax = ax)
end

function plotlv(T, group; size = (700, 350), shape, start = 1, 
        color = nothing, 
        #ellipse::Bool = false, prob = .95, circle::Bool = false
        zeros::Bool = false,
        xlabel = "", ylabel = "", title = "", kwargs...)
end

