function mbplsrda(Xbl, y; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    weights = mweight(ones(Q, n))
    mbplsrda(Xbl, y, weights; kwargs...)
end

function mbplsrda(Xbl, y, weights::Weight; kwargs...)
    res = dummy(y)
    ni = tab(y).vals
    fm = mbplsr(Xbl, res.Y, weights; kwargs...)
    Mbplsrda1(fm, res.lev, ni) 
end

function transf(object::Mbplsrda, Xbl; nlv = nothing)
    transf(object.fm, Xbl; nlv)
end

function predict(object::Mbplsrda, Xbl; nlv = nothing)
    Q = eltype(Xbl[1][1, 1])
    Qy = eltype(object.lev)
    m = nro(Xbl[1])
    a = nco(object.fm.T)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(Matrix{Qy}, le_nlv)
    posterior = list(Matrix{Q}, le_nlv)
    @inbounds for i = 1:le_nlv
        zpred = predict(object.fm, Xbl; nlv = nlv[i]).pred
        z =  mapslices(argmax, zpred; dims = 2)  # if equal, argmax takes the first
        pred[i] = reshape(replacebylev2(z, object.lev), m, 1)     
        posterior[i] = zpred
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end