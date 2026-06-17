"""
    occlknn(; kwargs...)
    occlknn(X; kwargs...)
One-class classification (OCC) using local kNN distance-based outlierness.
* `X` : Training X-data (n, p) assumed to represent the reference (= target) class.
Keyword arguments:
* `nsamp` : Nb. of observations (`X`-rows) sampled in the training data and for which are computed 
    the outliernesses (Monte Carlo simulation of the outlierness distribution of the reference class).
* `metric` : Metric used to compute the distances. See function `getknn`.
* `k` : Nb. nearest neighbors to consider.
* `algo` : Function summarizing the `k` distances to the neighbors.
* `typcut` : Type of cutoff. Possible values are: `:mad`, `:q`. See Thereafter.
* `cri` : When `typcut` = `:mad`, a constant. See thereafter.
* `alpha` : When `typcut` = `:q`, a risk-I level. See thereafter.
* `scal` : Symbol defining the column scaling of `X`. Possible values are: `:none`, `std` (uncorrected STD) and `prt` (pareto).

OCC using outlierness `d` as defined in function `outlknn`.

See function `occsd` for details on the cutoffs and outputs, and examples.

For predictions (`predict`), the outlierness of each new observation is compared to the outlierness 
distribution estimated from the `nsamp` observations sampled in the target class. 
""" 
occlknn(; kwargs...) = JchemoModel(occlknn, nothing, kwargs)

function occlknn(X; kwargs...)
    par = recovkw(ParOccknn{Q}, kwargs).par
    X = ensure_mat(X)
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
    k = min(par.k, n - 1)
    ## Distribution of outlierness of the 'nsamp' sampled training observations
    res = getknn(X, vX; k = k + 1, par.metric)
    d = similar(X, nsamp)
    nn = zeros(Int, k)
    @inbounds for i in eachindex(d)
        d[i] = par.algo(res.d[i][2:end])
        nn .= res.ind[i][2:end]
        res_nn = getknn(X, vrow(X, nn); k = k + 1, par.metric)
        d_nn = similar(X, k) 
        for j in eachindex(nn)
            d_nn[j] = par.algo(res_nn.d[j][2:end])
        end
        d[i] /= median(d_nn)
    end
    ## End 
    if par.typcut == :mad
        cutoff = median(d) + par.cri * madv(d)
    elseif par.typcut == :q
        cutoff = quantile(d, 1 - par.alpha)
    end
    e_cdf = StatsBase.ecdf(d)
    d = DataFrame(
        d = d, 
        dstand = d / cutoff, 
        pval = pval(e_cdf, d)
        )
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
    d = DataFrame(
        d = d, 
        dstand = d / object.cutoff, 
        pval = pval(object.e_cdf, d)
        )
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i in eachindex(d.d)]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

