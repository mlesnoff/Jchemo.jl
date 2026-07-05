""" 
    wtal(d::Vector{Q}; a::Q = 1.) where Q <: Float
Compute binary weights from distances using the 'talworth' distribution.
* `d` : A vector (n) of distances.
Keyword arguments:
* `a` : Parameter of the weight function, see below.

The returned weight vector w has component
* w[i] = 1 if |`d`[i]| <= `a`
* w[i] = 0 if |`d`[i]| > `a`

## Examples
```julia
d = rand(10)
wtal(d; a = .8)
```
""" 
function wtal(d::Vector{Q}; a::Q = 1.) where Q <: Float
    n = length(d)
    d = abs.(d)
    w = zeros(Q, n)
    w[d .<= a] .= 1
    w
end
