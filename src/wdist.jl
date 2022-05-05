"""
    wdist(d; h = 2, cri = 4, squared = false)
    wdist!(d; h = 2, cri = 4, squared = false)
Compute weights from distances, using a decreasing exponential function.
* `d` : A vector of distances.
* `h` : A scaling positive scalar defining the shape of the function. 
* `cri` : A positive scalar defining outliers in the distances vector.
* `squared`: If true, distances are replaced by the squared distances;
    the weight function is then a Gaussian (RBF) kernel function.

Weights are computed by exp(-d / (h * MAD(d))), or are set to 0 for 
distances > Median(d) + cri * MAD(d).

The weights decrease with increasing distances. Lower is h, sharper is the decreasing function. 
Weights are set to 0 for outliers (extreme distances).

    ## Examples
```julia
using CairoMakie, Distributions

x1 = rand(Chisq(10), 100) ;
x2 = rand(Chisq(40), 10) ;
d = [sqrt.(x1) ; sqrt.(x2)]
hist(d, nbins = 30)

h = 2 ; cri = 3
w = wdist(d; h = h, cri = cri) ;
hist(w, nbins = 30)
scatter(d, w)

d = collect(0:.5:15) ;
h = [.5, 1, 1.5, 2.5, 5, 10, Inf] ;
#h = [1, 2, 5, Inf] ;
w = wdist(d; h = h[1]) ;
f, ax = lines(d, w, label = string("h = ", h[1]))
for i = 2:length(h)
    w = wdist(d; h = h[i])
    lines!(ax, d, w, label = string("h = ", h[i]))
end
axislegend("Values of h")
f
```
"""  
function wdist(d; h = 2, cri = 4, squared = false)
    w = copy(d)
    wdist!(w; h = h, cri = cri, squared = squared)
    w
end

function wdist!(d; h = 2, cri = 4, squared = false)
    # d, out : (n,)
    squared ? d = d.^2 : nothing
    zmed =  Statistics.median(d)
    zmad = Jchemo.mad(d)
    cutoff = zmed + cri * zmad
    d .= map(x -> ifelse(x <= cutoff, exp(-x / (h * zmad)), zero(eltype(d))), d)
    d .= d / maximum(d)
    d[isnan.(d)] .= 1
    return
end
