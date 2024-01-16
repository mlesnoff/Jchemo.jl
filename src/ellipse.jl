## Working function building a 2-D ellipse
## Equation: (x - mu)' * S^(-1) * (x - mu) <= r^2
## * S : shape = variance-covariance matrix (size q x q) 
## of x (vector of length q)
## * mu : center (vector of length q)
## * radius = r
## * q = 2  
function ellipse(S; 
        mu = zeros(nco(S)), 
        radius = 1) 
    theta = collect(range(0, 2 * pi, length = 51))
    circ = radius * hcat(cos.(theta), sin.(theta))
    res = eigen(S; sortby = x -> -abs(x))
    d = sqrt.(res.values)
    V = res.vectors
    X = V * diagm(d) * circ'
    X = (X .+ mu)'
    (X = X, V, d)
end


