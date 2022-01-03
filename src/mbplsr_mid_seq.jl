struct MbplsrMidseq
    T::Matrix{Float64}
    C::Matrix{Float64}
    fm::Vector
    fmbl::Vector
    b::Vector
    ymeans::Vector{Float64}
end

function mbplsr_mid_seq(X, Y, weights = ones(size(X[1], 1)); nlv)
    nbl = length(X)
    zX = list(nbl, Matrix{Float64})
    @inbounds for i = 1:nbl
        zX[i] = copy(ensure_mat(X[i]))
    end
    Jchemo.mbplsr_mid_seq!(zX, copy(Y), weights; nlv = nlv)
end

function mbplsr_mid_seq!(X, Y, weights = ones(size(X[1], 1)); nlv)
    Y = ensure_mat(Y)
    n = size(X[1], 1)
    q = size(Y, 2)   
    nbl = length(X)
    weights = mweights(weights)
    D = Diagonal(weights)
    # Pre-allocation
    zT = similar(X[1], n, nbl)
    T = similar(X[1], n, nlv)
    C = similar(X[1], q, nlv)
    t   = similar(X[1], n)
    dt = copy(t)
    c = similar(X[1], q)
    # End
    fm = list(nlv)
    fmbl = list(nlv)
    b = list(nlv)
    for a = 1:nlv
        fmbl[a] = list(nbl)
        b[a] = list(nbl, Matrix{Float64})
        for i = 1:nbl
            fmbl[a][i] = plskern(X[i], Y, weights; nlv = 1)
            zT[:, i] .= fmbl[a][i].T[:, 1]
        end
        fm[a] = plskern(zT, Y, weights; nlv = 1) 
        t .= fm[a].T[:, 1]
        dt .= weights .* t
        tt = dot(t, dt)
        for i = 1:nbl
            b[a][i] = (t' * D) * X[i]  / tt
            X[i] .= X[i] .- t * b[a][i]
        end
        mul!(c, Y', dt)
        c ./= tt   
        T[:, a] .= t
        C[:, a] .= c
    end
    MbplsrMidseq(T, C, fm, fmbl, b, fm[1].ymeans)
end

function transform(object::MbplsrMidseq, X; nlv = nothing)
    m = size(X[i], 1)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    nbl = length(X)
    zT = similar(X[1], m, nbl)
    T = similar(X[1], m, nlv)
    zX = list(nbl, Matrix{Float64})
    @inbounds for i = 1:nbl
        zX[i] = copy(ensure_mat(X[i]))
    end
    for a = 1:nlv
        for i = 1:nbl
            zT[:, i] .= transform(object.fmbl[a][i], zX[i])
        end
        t = transform(object.fm[a], zT)
        for i = 1:nbl
            zX[i] .= zX[i] .- t * object.b[a][i]
        end
        T[:, a] .= t
    end
    T
end

function predict(object::MbplsrMidseq, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(le_nlv, Matrix{Float64})
    @inbounds for i = 1:le_nlv
        T = transform(object, X; nlv = nlv[i])
        pred[i] = object.ymeans' .+ T * object.C[:, 1:nlv[i]]'
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end



