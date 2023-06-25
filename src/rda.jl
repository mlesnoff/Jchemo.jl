
function rda(X, y; gamma, lb, prior = "unif")
    @assert gamma >= 0 && gamma <= 1 "gamma must be in [0, 1]"
    @assert lb >= 0 "lb must be in >= 0"
    X = ensure_mat(X)
    n, p = size(X)
    z = aggstat(X, y; fun = mean)
    ct = z.X
    lev = z.lev
    nlev = length(lev)
    res = matW(X, y)
    ni = res.ni
    if isequal(prior, "unif")
        wprior = ones(nlev) / nlev
    elseif isequal(prior, "prop")
        wprior = mweight(ni)
    end
    res.W .*= n / (n - nlev)
    Id = I(p)
    fm = list(nlev)
    @inbounds for i = 1:nlev
        ni[i] == 1 ? zn = n : zn = ni[i]
        res.Wi[i] .*= zn / (zn - 1)        
        @. res.Wi[i] = (1 - gamma) * res.Wi[i] + gamma * res.W
        #res.Wi[i] .= (1 - gamma) .* res.Wi[i] .+ gamma .* res.W
        res.Wi[i] .= res.Wi[i] .+ lb .* Id
        fm[i] = dmnorm(; mu = ct[i, :], S = res.Wi[i]) 
    end
    Qda(fm, res.Wi, ct, wprior, lev, ni)
end


