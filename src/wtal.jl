""" 
    wtal(d; a = 1)
Compute weights from distances using the 'talworth' distribution.
* `d` : Vector of distances.
Keyword arguments:
* `a` : Parameter of the weight function, 
    see below.

The returned weight vector w has component w[i] = 1 if |`d`[i]| <= `a`, 
and w[i] = 0 if |`d`[i]| > `a`.

## Examples
```julia
d = rand(10)
wtal(d; a = .8)
```
""" 
function wtal(d; a = 1)
    d = abs.(vec(d))
    Q = eltype(d)
    n = length(d)
    w = zeros(Q, n)
    w[d .<= a] .= 1
    w
end
