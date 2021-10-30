"""
    plotsp(X; color, kwargs...)
    Plotting spectra.
* `x` : X-data.
* `colors` : Set a unique color to the spectra.
* `kwargs` : Optional arguments to pass in `Axis`.
Plots lines corresponding to the rows of `x`.
""" 
function plotsp(X, wl = 1:size(X, 2); color = nothing, kwargs...) 
    n = size(X, 1)
    f = Figure()
    ax = Axis(f; kwargs...)
    @inbounds for i = 1:n
        if isnothing(color)
            lines!(ax, wl, X[i, :])
        else
            lines!(ax, wl, X[i, :]; color = color)
        end
    end
    f[1, 1] = ax
    f
end





