"""
    occstah(kwargs...)
    occstah(X; kwargs...)
One-class classification using the Stahel-Donoho outlierness.
* `X` : Training X-data (n, p).
Keyword arguments:
* `nlv` : Nb. dimensions on which `X` is projected. 
* `mcut` : Type of cutoff. Possible values are: `:mad`, 
    `:q`. See Thereafter.
* `cri` : When `mcut` = `:mad`, a constant. See thereafter.
* `risk` : When `mcut` = `:q`, a risk-I level. See thereafter.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled such as in function `stah`.

In this method, the outlierness `d` of a given observation
is the Stahel-Donoho outlierness (see `?stah`).

See function `occsd` for details on outputs, and examples. 
""" 
function occstah(X; kwargs...) 
    par = recovkwargs(Par, kwargs) 
    @assert 0 <= par.risk <= 1 "Argument 'risk' must ∈ [0, 1]."
    res = stah(X, par.nlv; 
        scal = par.scal)
    d = res.d
    #d2 = d.^2 
    #mu = median(d2)
    #s2 = mad(d2)^2
    #nu = 2 * mu^2 / s2
    #g = mu / nu
    #dist = Distributions.Chisq(nu)
    #pval = Distributions.ccdf.(dist, d2 / g)
    #mcut == :par ? cutoff = sqrt(g * quantile(dist, 1 - risk)) : nothing
    #mcut == "npar" ? cutoff = median(d) + par.cri * mad(d) : nothing  
    par.mcut == :mad ? cutoff = median(d) + 
        par.cri * mad(d) : nothing
    par.mcut == :q ? cutoff = quantile(d, 1 - par.risk) : 
        nothing
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, 
        pval = p_val)
    Occstah(d, res, e_cdf, cutoff)
end

"""
    predict(object::Occstah, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occstah, X)
    zX = copy(ensure_mat(X))
    m = nro(zX)
    res = object.res_stah
    fcenter!(zX, res.mu_scal)
    fscale!(zX, res.s_scal)
    T = zX * res.P
    fcenter!(T, res.mu)
    fscale!(T, res.s)
    T .= abs.(T)
    d = similar(T, m)
    @inbounds for i = 1:m
        d[i] = maximum(vrow(T, i))
    end
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, 
        dstand = d / object.cutoff, pval = p_val)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end


