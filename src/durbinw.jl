""" 
    durbinw(x::AbstractVector{Q}) where Q <: Float
Durbin-Watson statistic. 
* `x` : A vector (n).

The function returns the Durbin-Watson statistic, d, for a vector `x`. 
    
Value of d always lies between 0 and 4.
* For large n, d ~ 2(1 − ρ), where ρ is the sample autocorrelation of `x` at lag 1. 
    Value d = 2 therefore indicates no autocorrelation.  
* Values d substantially less than 2 (and particularlyclose to 0, or < 1) indicate that successive 
    elements of `x` are positively correlated. 
* If d > 2, successive elements of `x` are negatively correlated.

In chemometrics and for PLS models, d has for instance been applied to loading vectors, loading 
weights and b-coefficients to help to select the model dimensionnality (Rutledge & Barros, 2002).  

# References

https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic

Rutledge, D.N., Barros, A.S., 2002. Durbin–Watson statistic as a morphological estimator
of information content. Analytica Chimica Acta 454, 277–295. https://doi.org/10.1016/S0003-2670(01)01555-0

# Examples
```julia
using Jchemo

n = 10^3
x = randn(n)

durbinw(x)
```
"""
function durbinw(x::AbstractVector{Q}) where Q <: Float 
    v = [x[i] - x[i - 1] for i in 2:length(x)]
    norm2v(v) / norm2v(x)  # = dot(v, v) / dot(x, x)
end



