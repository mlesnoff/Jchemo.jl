struct Dmkern
    X::Array{Float64}
    H::Array{Float64}
    Hinv::Array{Float64}
    detH::Float64
end

function dmkern(X; H = nothing, a = .5)
    X = ensure_mat(X)
    n, p = size(X)
    ## Case where n = 1
    ## (ad'hoc for discrimination functions only)
    if n == 1
        H = diagm(repeat([a * n^(-1/(p + 4))], p))
    end
    ## End
    if isnothing(H)
        h = a * n^(-1 / (p + 4)) * colstd(X)      # a = .9, 1.06
        H = diagm(h)
    else 
        isa(H, Real) ? H = diagm(repeat([H], p)) : nothing
    end
    Hinv = inv(H)
    detH = det(H)
    detH == 0 ? detH = 1e-20 : nothing
    Dmkern(X, H, Hinv, detH)
end

function predict(object::Dmkern, X)
    X = ensure_mat(X)
    n, p = size(object.X)
    m = nro(X)
    pred = similar(X, m, 1)
    M = similar(object.X)
    @inbounds for i = 1:m
        M .= (vrow(X, i:i) .- object.X) * object.Hinv
        sum2 = rowsum(M.^2)
        pred[i, 1] = 1 / n * (2 * pi)^(-p / 2) * (1 / object.detH) * sum(exp.(-.5 * sum2))
    end
    (pred = pred,)
end



