""" 
    fweight(d; typw = "bisquare", alpha = 0)
Compute weights from distances and weight (e.g. bisquare) functions.
* `d` : Vector of distances.
* `typw` : Define the weight function.
* `alpha` : Parameter of the weight function.

The returned weight vector is: 
* w = f(`d` / q) where f is the weight function and q the 1-`alpha` 
quantile of `d` (Cleveland & Grosse 1991).

Possible values for `typw` are: 
* "bisquare": w = (1 - x^2)^2 
* "cauchy": w = 1 / (1 + x^2) 
* "epan": w = 1 - x^2 
* "fair": w =  1 / (1 + x)^2 
* "invexp": w = exp(-x) 
* "gauss": w = exp(-x^2)
* "trian": w = 1 - x  
* "tricube": w = (1 - x^3)^3  

## References
Cleveland, W.S., Grosse, E., 1991. Computational methods for local regression. 
Stat Comput 1, 47–62. https://doi.org/10.1007/BF01890836

## Examples
```julia
cols = cgrad(:tab10, collect(1:9)) ;
d = sort(sqrt.(rand(Chi(1), 1000)))
alpha = 0
typw = "bisquare"
w = fweight(d; typw = typw, alpha = alpha)
f, ax = lines(d, w, label = typw, color = cols[1],
    axis = (xlabel = "d", ylabel = "weight"))
typw = "cauchy"
w = fweight(d; typw = typw, alpha = alpha)
lines!(ax, d, w, label = typw, color = cols[2])
typw = "epan"
w = fweight(d; typw = typw, alpha = alpha)
lines!(ax, d, w, label = typw, color = cols[3])
typw = "fair"
w = fweight(d; typw = typw, alpha = alpha)
lines!(ax, d, w, label = typw, color = cols[4])
typw = "gauss"
w = fweight(d; typw = typw, alpha = alpha)
lines!(ax, d, w, label = typw, color = cols[5])
typw = "trian"
w = fweight(d; typw = typw, alpha = alpha)
lines!(ax, d, w, label = typw, color = cols[6])
typw = "invexp"
w = fweight(d; typw = typw, alpha = alpha)
lines!(ax, d, w, label = typw, color = cols[8])
typw = "tricube"
w = fweight(d; typw = typw, alpha = alpha)
lines!(ax, d, w, label = typw, color = cols[9])
axislegend("Function")
f
```
""" 
function fweight(d; typw = "bisquare", alpha = 0)
    d = vec(abs.(d))
    alpha = max(0, min(1, alpha))
    zd = d[isnan.(d) .== 0]
    q = quantile(zd, 1 - alpha)
    d ./= q
    typw == "bisquare" ? w = (1 .- d.^2).^2 : nothing 
    typw == "cauchy" ? w = 1 ./ (1 .+ d.^2) : nothing 
    typw == "epan" ? w = 1 .- d.^2 : nothing 
    typw == "fair" ? w =  1 ./ (1 .+ d).^2 : nothing 
    #if typw == "inv"
    #    w = 1 ./ d
    #    w ./= maximum(w[isnan.(w) .== 0])
    #end 
    typw == "invexp" ? w = exp.(-d) : nothing 
    typw == "gauss" ? w = exp.(-d.^2) : nothing
    typw == "trian" ? w = 1 .- d : nothing  
    typw == "tricube" ? w = (1 .- d.^3).^3 : nothing  
    w[d .> 1] .= 0
    w[isnan.(w)] .= 0 
    w
end

