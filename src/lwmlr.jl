struct Lwmlr2
    X::Array{Float64}
    Y::Array{Float64}
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

function lwmlr(X, Y; metric, 
        h, k, tol = 1e-4, verbose = false)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    Lwmlr2(X, Y, metric, h, k, tol, 
        verbose)
end

function predict(object::Lwmlr2, X)
    X = ensure_mat(X)
    m = nro(X)
    # Getknn
    res = getknn(object.X, X; k = object.k, 
            metric = object.metric)
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    # End
    pred = locw(object.X, object.Y, X; 
        listnn = res.ind, listw = listw, fun = mlr,
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end

