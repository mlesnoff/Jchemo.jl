function outlknn(X; metric = :eucl, k, algo = median, scal::Bool = false)
    outlknn!(copy(ensure_mat(X)); k, metric, algo, scal)
end

function outlknn!(X::Matrix; metric = :eucl, k, algo = median, scal::Bool = false)
    Q = eltype(X)
    n, p = size(X)
    xscales = ones(Q, p)
    if scal
        xscales .= colstd(X)
        fscale!(X, xscales)
    end
    k > n - 1 ? k = n - 1 : nothing
    res = getknn(X, X; k = k + 1, metric)
    d = zeros(n)
    nn = zeros(Int, k)
    @inbounds for i in eachindex(d)
        d[i] = algo(res.d[i][2:end])
        nn .= res.ind[i][2:end]
        res_nn = getknn(X, vrow(X, nn); k = k + 1, metric)
        d_nn = zeros(Q, k) 
        for j in eachindex(nn)
            d_nn[j] = median(res_nn.d[j][2:end])
        end
        d[i] /= median(d_nn)
    end
    (d = d, xscales)
end



