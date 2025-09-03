"""
    winvs(d; h = 2, criw = 4, squared = false)
    winvs!(d; h = 2, criw = 4, squared = false)
Compute weights from distances using an inverse scaled exponential function.
* `d` : A vector of distances.
Keyword arguments:
* `h` : A scaling positive scalar defining the shape of the weight function. 
* `criw` : A positive scalar defining outliers in the distances vector `d`.
* `squared`: If `true`, distances are replaced by the squared distances; the weight function 
    is then a Gaussian (RBF) kernel function.

Weights are computed by: 
* exp(-`d` / (`h` * MAD(`d`)))
or are set to 0 for extreme (potentially outlier) distances such as `d` > Median(`d`) + criw * MAD(`d`). 
Finally, weights are standardized to their maximal value. This is an adaptation of the weight function 
presented in Kim et al. 2011.

The weights decrease when distances increase. Lower is h, sharper is the decreasing function.

## References 

Kim S, Kano M, Nakagawa H, Hasebe S. Estimation of active pharmaceutical ingredients content using locally 
weighted partial least squares and statistical wavelength selection. Int J Pharm. 2011;
421(2):269-274. https://doi.org/10.1016/j.ijpharm.2011.10.007

## Examples
```julia
using Jchemo, CairoMakie, Distributions

x1 = rand(Chisq(10), 100) ;
x2 = rand(Chisq(40), 10) ;
d = [sqrt.(x1) ; sqrt.(x2)]
h = 2 ; criw = 3
w = winvs(d; h, criw) ;
f = Figure(size = (600, 300))
ax1 = Axis(f, xlabel = "Distance", ylabel = "Nb. observations")
hist!(ax1, d, bins = 30)
ax2 = Axis(f, xlabel = "Distance", ylabel = "Weight")
scatter!(ax2, d, w)
f[1, 1] = ax1 
f[1, 2] = ax2 
f

d = collect(0:.5:15) ;
h = [.5, 1, 1.5, 2.5, 5, 10, Inf] 
#h = [1, 2, 5, Inf] 
w = winvs(d; h = h[1]) 
f = Figure(size = (500, 400))
ax = Axis(f, xlabel = "Distance", ylabel = "Weight")
lines!(ax, d, w, label = string("h = ", h[1]))
for i = 2:length(h)
    w = winvs(d; h = h[i])
    lines!(ax, d, w, label = string("h = ", h[i]))
end
axislegend("Values of h"; position = :lb)
f[1, 1] = ax
f
```
"""  
function winvs(d; h = 2, criw = 4, squared = false)
    w = copy(d)
    winvs!(w; h, criw, squared = squared)
    w
end

function winvs!(d; h = 2, criw = 4, squared = false)
    squared ? d .= d.^2 : nothing
    zmed =  Statistics.median(d)
    zmad = madv(d)
    cutoff = zmed + criw * zmad
    d .= map(x -> ifelse(x <= cutoff, exp(-x / (h * zmad)), zero(eltype(d))), d)
    ## Alternative, e.g.: 
    ## d .= wdis(d; typw = :bisquare)
    d .= d / maximum(d)
    d[isnan.(d)] .= 1
    return
end
