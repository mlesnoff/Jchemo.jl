function outknn(X; metric = :eucl, k, algo = median, scal::Bool = false)
    outknn!(copy(ensure_mat(X)); k, metric, algo, scal)
end

function outknn!(X::Matrix; metric = :eucl, k, algo = median, scal::Bool = false)
    Q = eltype(X)
    n, p = size(X)
    xscales = ones(Q, p)
    if scal
        xscales .= colstd(X)
        fscale!(X, xscales)
    end
    res = getknn(X, X; k = k + 1, metric)
    d = zeros(n)
    @inbounds for i in eachindex(d)
        d[i] = algo(res.d[i][2:end])
    end
    (d = d, xscales)
end



