# Build a 2D ellipse
# Equation: (x - mu)' * S^(-1) * (x - mu) <= r^2
# shape = variance-covariance matrix S (size q x q) of x (vector of length q)
# center = mu (vector of length q)
# radius = r
# q = 2  
function ellipse(shape; center = zeros(nco(shape)), radius = 1) 
    theta = collect(range(0, 2 * pi, length = 51))
    circ = radius * hcat(cos.(theta), sin.(theta))
    res = eigen(shape; sortby = x -> -abs(x))
    d = sqrt.(res.values)
    V = res.vectors
    X = V * diagm(d) * circ'
    X = (X .+ center)'
    (X = X, V, d)
end


