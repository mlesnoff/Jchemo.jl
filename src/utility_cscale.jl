"""
    center(X, v)
    center!(X::AbstractMatrix, v)
Center each column of `X`.
* `X` : Data.
* `v` : Centering factors.

## examples
```julia
n, p = 5, 6
X = rand(n, p)
xmeans = colmean(X)
center(X, xmeans)
```
""" 
center(X, v) = X .- vec(v)'

function center!(X::AbstractMatrix, v)
    X .-= vec(v)'
end

#function center(X, v)
#    zX = copy(ensure_mat(X))
#    center!(zX, v)
#    zX
#end
#function center!(X::AbstractMatrix, v)
#    p = nco(X)
#    @inbounds for i = 1:p
#        X[:, i] .= vcol(X, i) .- v[i]
#    end
#end

"""
    scale(X, v)
    scale!(X::AbstractMatrix, v)
Scale each column of `X`.
* `X` : Data.
* `v` : Scaling factors.

## Examples
```julia
X = rand(5, 2) 
scale(X, colstd(X))
```
""" 

scale3(X, v) = X ./ vec(v)'

function scale3!(X::AbstractMatrix, v)
    X ./= vec(v)'
end


function scale(X, v)
    zX = copy(ensure_mat(X))
    scale!(zX, v)
    zX
end

function scale!(X::AbstractMatrix, v)
    p = nco(X)
    @inbounds for i = 1:p
        X[:, i] .= vcol(X, i) ./ v[i]
    end
end

# Below: Much slower and requires more memories
scale2(X, v) = mapslices(function f(x) ; x ./ v ; end, ensure_mat(X), dims = 2)



"""
    cscale(X, u, v)
    cscale!(X, u, v)
Center and scale each column of `X`.
* `X` : Data.
* `u` : Centering factors.
* `v` : Scaling factors.

## examples
```julia
n, p = 5, 6
X = rand(n, p)
xmeans = colmean(X)
xstds = colstd(X)
cscale(X, xmeans, xstds)
```
""" 
function cscale(X, u, v)
    zX = copy(ensure_mat(X))
    cscale!(zX, u, v)
    zX
end

function cscale!(X::AbstractMatrix, u, v)
    p = nco(X)
    @inbounds for i = 1:p
        X[:, i] .= (vcol(X, i) .- u[i]) ./ v[i]
    end
end

# Slower:
function cscale2!(X::AbstractMatrix, u, v)
    center!(X, u)
    scale!(X, v)
end


