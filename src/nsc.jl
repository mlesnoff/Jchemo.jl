struct Nsc
    ds::Array{Float64}
    cts::Array{Float64}
    d::Array{Float64}
    ct::Array{Float64}
    sel::Vector{Int64}
    poolstd::Vector{Float64}
    s0::Real
    mi::Vector{Float64}
    ni::Vector{Int64}
    lev::AbstractVector
    theta::Vector{Float64}
    delta::Real
    xscales::Vector{Float64}
    weights::Vector{Float64}
end

function nsc(X, y, weights = ones(nro(X)); 
        delta = .5, scal = false)
    X = ensure_mat(X)
    y = vec(y)    # for findall
    n, p = size(X)
    weights = mweight(weights)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X, weights)
        X = scale(X, xscales)
    end
    xmeans = colmean(X, weights)
    taby = tab(y)
    lev = taby.keys
    ni = taby.vals
    nlev = length(lev)
    theta = vec(aggstat(weights, y; fun = sum).X)
    ct = similar(X, nlev, p)
    d = copy(ct)
    poolstd = zeros(p)
    mi = similar(X, nlev)
    @inbounds for i = 1:nlev
        s = y .== lev[i]
        ct[i, :] .= colmean(X[s, :], weights[s])
        poolstd .= poolstd .+ theta[i] .* colvar(X[s, :], weights[s])
        d[i, :] .= ct[i, :] .- xmeans
        mi[i] = sqrt(1 / ni[i] - 1 / n)
    end
    poolstd .= sqrt.(poolstd * n / (n - nlev))   # Pooled within-class stds
    s0 = median(poolstd)
    poolstd_s0 = poolstd .+ s0
    scale!(d, poolstd_s0)
    d ./= mi
    ds = soft.(d, delta)
    cts = scale(ds, 1 ./ poolstd_s0) .* mi
    sel = findall(colsum(abs.(ds)) .> 0)
    Nsc(ds, cts, d, ct, sel, poolstd, s0, mi,
        ni, lev, theta, delta, xscales, weights)
end
