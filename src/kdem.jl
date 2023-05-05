struct Kdem
    X::Array{Float64}
    H::Array{Float64}
    Hinv::Array{Float64}
    detH::Float64
end

function kdem(X; H = nothing, h = nothing, a = .5)
    X = ensure_mat(X)
    n, p = size(X)
    ## Case where n = 1 (for discrimination functions)
    if n == 1
        H = diagm(repeat([a * n^(-1/(p + 4))], p))
    end
    ## End
    if isnothing(H)
        if !isnothing(h)
            isa(h, Real) ? H = diagm(repeat([h], p)) : H = diagm(h)
        else
            h = a * n^(-1 / (p + 4)) * colstd(X)      # a = .9, 1.06
            H = diagm(h)
        end
    end
    Hinv = inv(H)
    detH = det(H)
    detH == 0 ? detH = 1e-20 : nothing
    (X, H, Hinv, detH)
end

function predict(object::Kdem, X)
    X = ensure_mat(X)
    n, p = size(object.X)
    m = nro(X)
    pred = similar(X, m, 1)
    M = similar(X)
    @inbounds for i = 1:m
        M .= (vrow(X, i:i) .- object.X) * object.Hinv
        sum2 = rowsum(M.^2)
        pred[i, 1] = 1 / n * (2 * pi)^(-p / 2) * (1 / object.detH) * sum(exp.(-.5 * sum2))
    end
    (pred = pred,)
end



