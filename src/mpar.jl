"""
    mpar(; kwargs...)
Return a named tuple with all the combinations of the parameter values defined in kwargs.
Keyword arguments:
* `kwargs` : Named vector(s) of the parameter(s) values.

## Examples
```julia
using Jchemo
nlvdis = 25 ; metric = [:mah] 
h = [1 ; 2 ; Inf] ; k = [500 ; 1000] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k) 
length(pars[1])
reduce(hcat, pars)
```
"""
mpar = function(; kwargs...)
    iter = Base.product(values(kwargs)...)
    nam = [a.first for a in kwargs]
    Jchemo.mpar_work(iter, nam)
end


## Not exported 

mpar_tupl = function(tupl::NamedTuple)
    iter = Base.product(values(tupl)...)
    nam = @names tupl  
    Jchemo.mpar_work(iter, nam)
end

mpar_work = function(iter::Base.Iterators.ProductIterator, nam::Union{Vector, Tuple})
    z = collect(iter) # matrix (n, 1)
    p = length(z[1])
    u = collect(Iterators.flatten(z))
    u = reshape(u, p, :) # matrix (p, n)
    v = ntuple(i -> u[i, :], p)
    v = (; zip(nam, v)...)
    v
end


