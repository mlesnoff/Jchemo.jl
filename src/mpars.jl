"""
    mpars(; kwargs...)
Returns a tuple with all the combinations
of the parameters defined in kwargs.
kwargs: vectors of same size.
"""
mpars = function(; kwargs...)
    nam = [a.first for a in kwargs]
    iter = Base.product(values(kwargs)...)
    z = vec(collect(iter))
    u = collect(zip(z...)) 
    v = collect.(u)
    v = (; zip(nam, v)...)
end

