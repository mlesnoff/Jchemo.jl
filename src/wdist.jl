"""
    wdist(d, h; cri = 4, squared = false)
Compute weights from distances, using a decreasing exponential function.
* `d` : A vector (n,) of distances.
* `h` : A scaling positive scalar defining the shape of the function. 
* `cri` : A positive scalar defining outliers in the distances vector.
* `squared`: If true, distances are replaced by the squared distances;
the weight function is then a Gaussian (RBF) kernel function.

Computed weights decrease with increasing distances. Lower is h, sharper is the decreasing function. 
Weights are set to 0 for distance outliers.

The in-place function stores the output in `d`.
"""  
function wdist(d, h; cri = 4, squared = false)
    w = copy(d)
    wdist!(w, h; cri = cri, squared = squared)
    w
end

function wdist!(d, h; cri = 4, squared = false)
    ## d, out : (n,)
    squared ? d = d.^2 : nothing
    zmed =  Statistics.median(d)
    zmad = Jchemo.mad(d)
    cutoff = zmed + cri * zmad
    d .= map(x -> ifelse(x <= cutoff, exp(-x / (h * zmad)), zero(eltype(d))), d)
    d .= d ./ maximum(d)
    d[isnan.(d)] .= 1
    return
end
