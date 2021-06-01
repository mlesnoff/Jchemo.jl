wdist = function(d, h ; cri = 4, squared = false)
    squared ? d = d.^2 : nothing
    zmed =  median(d)
    zmad = Jchemo.mad(d)
    cutoff = zmed + cri * zmad
    w = map(x -> ifelse(x <= cutoff, exp(-x / (h * zmad)), zero(eltype(d))), d)
    w = w / maximum(w)
    w[isnan.(w)] .= 1
    w
end




