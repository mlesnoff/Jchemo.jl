"""
    occlknn(; kwargs...)
    occlknn(X; kwargs...)
One-class classification using local kNN distance-based outlierness.
* `X` : Training X-data (n, p) assumed to represent the reference class.
Keyword arguments:
* `nsamp` : Nb. of observations (`X`-rows) sampled in the training data and for which are computed 
    the outliernesses (stimated outlierness distribution of the reference class).
* `metric` : Metric used to compute the distances. See function `getknn`.
* `k` : Nb. nearest neighbors to consider.
* `algo` : Function summarizing the `k` distances to the neighbors.
* `cut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `cut` = `:mad`, a constant. See thereafter.
* `risk` : When `cut` = `:q`, a risk-I level. See thereafter.
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

See functions:
* `occknn` for examples,
* `outlknn` for details on the outlierness computation method,
* and `occsd` for details on the the cutoff computation and the outputs.

For **predictions** (`predict`), the outlierness of each new observation is compared to the outlierness 
distribution estimated from the `nsamp` sampled observations. 
""" 
occlknn(; kwargs...) = JchemoModel(occlknn, nothing, kwargs)

function occlknn(X; kwargs...)
    par = recovkw(ParOccknn, kwargs).par
    X = ensure_mat(X)
    Q = eltype(X)
    n, p = size(X)
    xscales = ones(Q, p)
    if par.scal
        xscales .= colstd(X)
        X = fscale(X, xscales)
    end
    nsamp = min(par.nsamp, n)
    if nsamp == n
        s = 1:n
    else
        s = sample(1:n, nsamp, replace = false)
    end
    vX = vrow(X, s)
    par.k > n - 1 ? k = n - 1 : k = par.k
    metric = par.metric
    algo = par.algo
    ## kNN distance
    res = getknn(X, vX; k = k + 1, metric)
    d = similar(X, nsamp)
    nn = zeros(Int, k)
    @inbounds for i in eachindex(d)
        d[i] = algo(res.d[i][2:end])
        nn .= res.ind[i][2:end]
        res_nn = getknn(X, vrow(X, nn); k = k + 1, metric)
        d_nn = similar(X, k) 
        for j in eachindex(nn)
            d_nn[j] = algo(res_nn.d[j][2:end])
        end
        d[i] /= median(d_nn)
    end
    ## End 
    par.cut == :mad ? cutoff = median(d) + par.cri * madv(d) : nothing
    par.cut == :q ? cutoff = quantile(d, 1 - par.risk) : nothing
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val)
    Occlknn(d, X, e_cdf, cutoff, xscales, par)
end

function predict(object::Occlknn, X)
    X = ensure_mat(X)
    m = nro(X)
    k = object.par.k
    metric = object.par.metric
    algo = object.par.algo
    ## kNN distance
    res = getknn(object.X, fscale(X, object.xscales); k = k + 1, metric)
    d = similar(X, m)
    nn = zeros(Int, k)
    @inbounds for i in eachindex(d)
        d[i] = algo(res.d[i][2:end])
        nn .= res.ind[i][2:end]
        res_nn = getknn(object.X, vrow(object.X, nn); k = k + 1, metric)
        d_nn = similar(X, k) 
        for j in eachindex(nn)
            d_nn[j] = algo(res_nn.d[j][2:end])
        end
        d[i] /= median(d_nn)
    end
    ## End
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = p_val)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

