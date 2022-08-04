"""
    mpar(; kwargs...)
Return a tuple with all the combinations of the parameter values defined in kwargs.
* `kwargs` : vector(s) of the parameter(s) values.

## Examples
```julia
nlvdis = 25 ; metric = ["mahal"] 
h = [1 ; 2 ; Inf] ; k = [500 ; 1000] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k) 
length(pars[1])
reduce(hcat, pars)
```
"""
mpar = function(; kwargs...)
    nam = [a.first for a in kwargs]
    iter = Base.product(values(kwargs)...)
    z = collect(iter) # matrix (n, 1)
    p = length(z[1])
    u = collect(Iterators.flatten(z))
    u = reshape(u, p, :) # matrix (p, n)
    v = ntuple(i -> u[i, :], p)
    v = (; zip(nam, v)...)
end

