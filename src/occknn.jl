Base.@kwdef mutable struct ParOccknn1
    nsamp::Int = 100
    metric::Symbol = :eucl                                       
    k::Int = 1     
    algo::Function = sum
    cut::Symbol = :mad   
    cri::Float64 = 3.
    risk::Float64 = .025 
    scal::Bool = false                
end 

struct Occknn1
    d::DataFrame
    X::Matrix
    e_cdf::ECDF
    cutoff::Real
    xscales::Vector
    par::ParOccknn1
end

occknn(; kwargs...) = JchemoModel(occknn, nothing, kwargs)

function occknn(X; kwargs...)
    par = recovkw(ParOccknn1, kwargs).par
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
    ## kNN distance
    par.k > n - 1 ? k = n - 1 : k = par.k
    res = getknn(X, vX; k = k + 1, metric = par.metric)
    d = zeros(nsamp)
    @inbounds for i in eachindex(d)
        d[i] = par.algo(res.d[i][2:end])
    end
    ## End 
    par.cut == :mad ? cutoff = median(d) + par.cri * madv(d) : nothing
    par.cut == :q ? cutoff = quantile(d, 1 - par.risk) : nothing
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val)
    Occknn1(d, X, e_cdf, cutoff, xscales, par)
end

function predict(object::Occknn1, X)
    X = ensure_mat(X)
    m = nro(X)
    ## kNN distance
    res = getknn(object.X, fscale(X, object.xscales); k = object.par.k + 1, metric = object.par.metric) 
    d = similar(X, m)
    @inbounds for i in eachindex(d)
        d[i] = object.par.algo(res.d[i][2:end])
    end
    ## End
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = p_val)
    pred = [if d.dstand[i] <= 1 "in" else "out" end for i = 1:m]
    pred = reshape(pred, m, 1)
    (pred = pred, d)
end

