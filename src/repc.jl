"""
    repc(X::AbstractMatrix{Q}, group::Vector{String}) where Q <: Float
    repc!(X::Matrix{Q}, group::Vector{String}) where Q <: Float
Center the rows of a matrix by groups.
* `X` : Data (n, p).
* `group` : A variable (n) representing the group membership. Must be a `Vector{String}`.

Each sub-matrix of `X` corresponding to a group is centered by its column mean. 

## Examples
```julia
@head X = rand(15, 3)
group = rand(["A", "B", "C"], 15)
@head Xc = repc(X, group)
i = 1 ; colmean(Xc[group .== mlev(group)[i], :])
colmean(Xc)
```
""" 
function repc(X::AbstractMatrix{Q}, group::Vector{String}) where Q <: Float
    zX = ensure_mat(copy(X))
    repc!(zX, group)
    zX
end

function repc!(X::Matrix{Q}, group::Vector{String}) where Q <: Float
    lev = mlev(group)
    xmeans = similar(X, nco(X))
    @inbounds for i in eachindex(lev)
        s = group .== lev[i]
        xmeans .= colmean(vrow(X, s))
        X[s, :] .-= xmeans'
    end
end
