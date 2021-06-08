
"""
snv(X)
snv!(X)
Standard normal variate transformation
""" 
snv(X) = snv!(copy(X))
function snv!(X) 
    X .= (X .- Statistics.mean(X; dims = 2)) ./ Statistics.std(X; corrected = false, dims = 2)
end

"""
fdif(X)
fdif!(X, r)
fdif2(X, r)
Finite difference method for computing 1st and 2nd discrete derivatives
- r: range (nb. intervals of two successive colums) for the finite differences
""" 
fdif(X, r = 5) = fdif!(copy(X), r)
fdif2(X, r = 5) = fdif!(fdif!(copy(X), r))
function fdif!(X, r = 5)  
    p = size(X, 2)
    zp = p - r
    @inbounds for j = 1:zp
        X[:, j] .= col(X, j + r) .- col(X, j)
    end
    @view(X[:, 1:zp])
end




