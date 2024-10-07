""" 
    fweight(d; typw = :bisquare, alpha = 0)
Computation of weights from distances.
* `d` : Vector of distances.
Keyword arguments:
* `typw` : Define the weight function.
* `alpha` : Parameter of the weight function, 
    see below.

The returned weight vector is: 
* w = f(`d` / q) where f is the weight function 
    and q the 1-`alpha` quantile of `d` 
    (Cleveland & Grosse 1991).

Possible values for `typw` are: 
* :bisquare: w = (1 - d^2)^2 
* :cauchy: w = 1 / (1 + d^2) 
* :epan: w = 1 - d^2 
* :fair: w =  1 / (1 + d)^2 
* :invexp: w = exp(-d) 
* :invexp2: w = exp(-d / 2) 
* :gauss: w = exp(-d^2)
* :trian: w = 1 - d  
* :tricube: w = (1 - d^3)^3  

## References
Cleveland, W.S., Grosse, E., 1991. Computational methods for local regression. 
Stat Comput 1, 47–62. https://doi.org/10.1007/BF01890836

## Examples
```julia
using Jchemo, CairoMakie, Distributions

d = sort(sqrt.(rand(Chi(1), 1000)))
colm = cgrad(:tab10, collect(1:9)) ;
alpha = 0
f = Figure(size = (600, 500))
ax = Axis(f, xlabel = "d", ylabel = "Weight")
typw = :bisquare
w = fweight(d; typw, alpha)
lines!(ax, d, w, label = String(typw), color = colm[1])
typw = :cauchy
w = fweight(d; typw, alpha)
lines!(ax, d, w, label = String(typw), color = colm[2])
typw = :epan
w = fweight(d; typw, alpha)
lines!(ax, d, w, label = String(typw), color = colm[3])
typw = :fair
w = fweight(d; typw, alpha)
lines!(ax, d, w, label = String(typw), color = colm[4])
typw = :gauss
w = fweight(d; typw, alpha)
lines!(ax, d, w, label = String(typw), color = colm[5])
typw = :trian
w = fweight(d; typw, alpha)
lines!(ax, d, w, label = String(typw), color = colm[6])
typw = :invexp
w = fweight(d; typw, alpha)
lines!(ax, d, w, label = String(typw), color = colm[7])
typw = :invexp2
w = fweight(d; typw, alpha)
lines!(ax, d, w, label = String(typw), color = colm[8])
typw = :tricube
w = fweight(d; typw, alpha)
lines!(ax, d, w, label = String(typw), color = colm[9])
axislegend("Function", position = :lb)
f[1, 1] = ax
f
```
""" 
function fweight(d; typw = :bisquare, alpha = 0)
    d = vec(abs.(d))
    alpha = max(0, min(1, alpha))
    zd = d[isnan.(d) .== 0]
    q = quantile(zd, 1 - alpha)  # = max when alpha = 0
    d ./= q                      # normalization (d = 1 for max when alpha = 0 ==> w = 0)
    typw == :bisquare ? w = (1 .- d.^2).^2 : nothing 
    typw == :cauchy ? w = 1 ./ (1 .+ d.^2) : nothing 
    typw == :epan ? w = 1 .- d.^2 : nothing 
    typw == :fair ? w =  1 ./ (1 .+ d).^2 : nothing 
    #if typw == :inv
    #    w = 1 ./ d
    #    w ./= maximum(w[isnan.(w) .== 0])
    #end 
    typw == :invexp ? w = exp.(-d) : nothing
    typw == :invexp2 ? w = exp.(-d / 2) : nothing  
    typw == :gauss ? w = exp.(-d.^2) : nothing
    typw == :trian ? w = 1 .- d : nothing  
    typw == :tricube ? w = (1 .- d.^3).^3 : nothing  
    w[d .> 1] .= 0
    w[isnan.(w)] .= 0 
    w
end

""" 
    talworth(d; a = 1)
Computation of weights from distances.
* `d` : Vector of distances.
Keyword arguments:
* `a` : Parameter of the weight function, 
    see below.

The returned weight vector w has component w[i] = 1 if `|d[i]| <= a`, 
and w[i] = 0 if `|d[i]| > a`.

## Examples
```julia
d = rand(10)
talworth(d; a = .8)
```
""" 
function talworth(d; a = 1)
    d = abs.(vec(d))
    Q = eltype(d)
    n = length(d)
    w = zeros(Q, n)
    w[d .<= a] .= 1
    w
end
