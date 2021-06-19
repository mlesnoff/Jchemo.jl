"""
    mpars(; kwargs...)
Returns a tuple with all the combinations
of the parameter values defined in kwargs.
- kwargs : vector(s) of the parameter(s) values.

Vectors in kwargs must be of same length.
"""
mpars = function(; kwargs...)
    nam = [a.first for a in kwargs]
    iter = Base.product(values(kwargs)...)
    z = vec(collect(iter))
    u = collect(zip(z...)) 
    v = collect.(u)
    v = (; zip(nam, v)...)
end

