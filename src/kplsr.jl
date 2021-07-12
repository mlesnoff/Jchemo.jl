struct Kplsr
    X::Array{Float64}
    Kt::Array{Float64}
    T::Array{Float64}
    C::Array{Float64}
    U::Array{Float64}
    R::Array{Float64}
    D::Array{Float64} 
    DKt::Array{Float64}
    vtot::Array{Float64}   
    ymeans::Vector{Float64}
    weights::Vector{Float64}
    kern
    dots
    iter::Vector{Int}
end

"""
    kplsr(X, Y, weights = ones(size(X, 1)); nlv , kern = "krbf", kwargs...)
Kernel partial least squares regression (KPLSR) implemented with a NIPALS algorithm (Rosipal & Trejo, 2001).

* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `nlv` : Nb. latent variables (LVs) to consider. 
* 'kern' : Type of kernel used to compute the Gram matrices.
    Possible values are "krbf" of "kpol" (see respective functions `krbf` and `kpol`).
* `kwargs` : Named arguments to pass in the kernel function.

This algorithm becomes slow for n > 1000.

The kernel Gram matrices are internally centered. 

## References 

Rosipal, R., Trejo, L.J., 2001. Kernel Partial Least Squares Regression in Reproducing Kernel Hilbert Space. 
Journal of Machine Learning Research 2, 97-123.

""" 
function kplsr(X, Y, weights = ones(size(X, 1)); nlv, kern = "krbf", tol = 1.5e-8, maxit = 100, kwargs...)
    kplsr!(copy(X), copy(Y), weights; nlv = nlv, kern = kern, tol = tol, maxit = maxit, kwargs...)
end

function kplsr!(X, Y, weights = ones(size(X, 1)); nlv, kern = "krbf", tol = 1.5e-8, maxit = 100, kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    q = size(Y, 2)
    weights = mweights(weights)
    ymeans = colmeans(Y, weights)   
    center!(Y, ymeans)  
    fkern = eval(Meta.parse(kern))  
    K = fkern(X, X; kwargs...)     # In the future: fkern!(K, X, X; kwargs...)
    D = Diagonal(weights)
    Kt = K'    
    DKt = D * Kt
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- vtot .+ sum(D * DKt')
    # Pre-allocation
    K = copy(Kc)
    T = similar(X, n, nlv)
    U = copy(T)
    C = similar(X, q, nlv)
    I = Diagonal(ones(n))
    iter = Int.(zeros(nlv))
    # temporary results
    t  = similar(X, n)
    dt = similar(X, n)
    c  = similar(X, q)
    u  = similar(X, n)
    zu = similar(X, n)
    # End
    for a in 1:nlv
        if q == 1      
            mul!(t, K, D * vec(Y))   # t = K * D * Y
            t ./= sqrt(dot(t, weights .* t))
            dt .= weights .* t
            mul!(c, Y', dt)
            u .= Y * c 
            u ./= sqrt(dot(u, u))
        else
            u .= Y[:, 1]
            ztol = 1.
            ziter = 1
            while ztol > tol && ziter <= maxit
                mul!(t, K, weights .* u)
                t ./= sqrt(dot(t, weights .* t))
                dt .= weights .* t                
                mul!(c, Y', dt)
                zu .= Y * c 
                zu ./= sqrt(dot(zu, zu))
                ztol = sqrt(sum((u - zu).^2))
                u .= zu
                ziter = ziter + 1
            end
            iter[a] = ziter - 1
        end
        z = I - t * dt'  # slow
        K .= z * K * z'  # slow
        Y .-= t * c'
        T[:, a] .= t
        C[:, a] .= c
        U[:, a] .= u
    end
    DU = D * U
    zR = DU * inv(T' * D * Kc * DU)
    Kplsr(X, Kt, T, C, U, zR, D, DKt, vtot, ymeans, weights, kern, kwargs, iter)
end

""" 
    transform(object::Kplsr, X; nlv = nothing)
Compute LVs (score matrix "T") from a fitted model and X-data.
* `object` : The maximal fitted model.
* `X` : X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider. If nothing, it is the maximum nb. LVs.
""" 
function transform(object::Kplsr, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    fkern = eval(Meta.parse(object.kern))
    K = fkern(X, object.X; object.dots...)
    DKt = object.D * K'
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- object.vtot .+ sum(object.D * object.DKt')
    T = Kc * @view(object.R[:, 1:nlv])
    T
end

"""
    coef(object::Kplsr; nlv = nothing)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
    If nothing, it is the maximum nb. LVs.
""" 
function coef(object::Kplsr; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    beta = object.C[:, 1:nlv]'
    q = length(object.ymeans)
    int = reshape(object.ymeans, 1, q)
    (int = int, beta = beta)
end

"""
    predict(object::Kplsr, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The maximal fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
If nothing, it is the maximum nb. LVs.
""" 
function predict(object::Kplsr, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    T = transform(object, X)
    pred = list(le_nlv)
    @inbounds for i = 1:le_nlv
        z = coef(object; nlv = nlv[i])
        pred[i] = z.int .+ @view(T[:, 1:nlv[i]]) * z.beta
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end
