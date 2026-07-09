function repc(X::AbstractMatrix{Q}, group::Vector{String}) where Q <: Jchemo.Float
    zX = copy(X)
    lev = mlev(group)
    xmeans = similar(X, nco(X))
    @inbounds for i in eachindex(lev)
        s = group .== lev[i]
        xmeans .= colmean(vrow(X, s))
        zX[s, :] .-= xmeans'
    end
    zX
end
