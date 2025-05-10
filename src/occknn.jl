struct Occknn1
    d::DataFrame
    fm
    T::Matrix
    tscales::Vector
    k::Int
    e_cdf::ECDF
    cutoff::Real    
    kwargs::Base.Pairs
    par::Nothing #Par
end

struct Occlknn1
    d::DataFrame
    fm
    T::Matrix
    tscales::Vector
    k::Int
    e_cdf::ECDF
    cutoff::Real    
    kwargs::Base.Pairs
    par::Nothing #Par
end

Base.@kwdef mutable struct ParOccknn1
    nsamp::Int = 100
    metric::Symbol = :eucl                                       
    k::Int = 1     
    algo::Function = sum
    cut::Symbol = :mad   
    risk::Float64 = .025  
    cri::Float64 = 3. 
    scal::Bool = false                    
end 

occknn(; kwargs...) = JchemoModel(occknn, nothing, kwargs)

function occknn(X; kwargs...)
    par = recovkw(ParOccknn1, kwargs).par
    X = ensure_mat(X)
    n = nro(X)
    nsamp = min(par.nsamp, n)
    if nsamp == n
        s = 1:n
    else
        s = sample(1:n, nsamp, replace = false)
    end
    vX = vrow(X, s)
    d = outknn(vX; metric = par.metric, k = par.k, algo = par.algo, scal = par.scal).d
    par.cut == :mad ? cutoff = median(d) + par.cri * madv(d) : nothing
    par.cut == :q ? cutoff = quantile(d, 1 - par.risk) : nothing
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val)
    (d = d, k = par.k, e_cdf, cutoff)
end


function predict(object::Occknn1, X)
    X = ensure_mat(X)
    m = nro(X)
    T = transf(object.fm, X)
    fscale!(T, object.tscales)
    res = getknn(object.T, T; k = object.k, metric = :eucl) 
    d = zeros(m)
    @inbounds for i = 1:m
        d[i] = median(res.d[i])
    end
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, pval = p_val)
    pred = reshape(Int.(d.dstand .> 1), m, 1)
    (pred = pred, d)
end




