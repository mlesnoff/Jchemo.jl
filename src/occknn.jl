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

function occknndis(X; nsamp, 
        metric = :eucl, k, algo = sum, scal::Bool = false, 
        typc = :mad, cri = 3, alpha = .025, 
        kwargs...
        )
    X = ensure_mat(X)
    n = nro(X)
    zX = 
    
    
    

    typc == :mad ? cutoff = median(d) + par.cri * mad(d) : nothing
    typc == :q ? cutoff = quantile(d, 1 - alpha) : nothing
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val)
    Occknn1(d, fm, fm.T, tscales, k, e_cdf, cutoff)
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




