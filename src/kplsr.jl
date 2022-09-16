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
    kplsr(X, Y, weights = ones(size(X, 1)); 
        nlv, kern = "krbf", tol = 1.5e-8, maxit = 100, kwargs...)
    kplsr!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); 
        nlv, kern = "krbf", tol = 1.5e-8, maxit = 100, kwargs...)
Kernel partial least squares regression (KPLSR) implemented with a NIPALS algorithm (Rosipal & Trejo, 2001).

* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations.
* `nlv` : Nb. latent variables (LVs) to consider. 
* 'kern' : Type of kernel used to compute the Gram matrices.
    Possible values are "krbf" or "kpol" (see respective functions `krbf` and `kpol`).
* `tol` : Tolerance value for stopping the iterations.
* `maxit` : Maximum nb. iterations.
* `kwargs` : Named arguments to pass in the kernel function.

This algorithm becomes slow for n > 1000.

The kernel Gram matrices are internally centered. 

## References 
Rosipal, R., Trejo, L.J., 2001. Kernel Partial Least Squares Regression in Reproducing Kernel Hilbert Space. 
Journal of Machine Learning Research 2, 97-123.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 20 ; gamma = 1e-1
fm = kplsr(Xtrain, ytrain; nlv = nlv, gamma = gamma) ;
fm.T

zcoef = coef(fm)
zcoef.int
zcoef.beta
coef(fm; nlv = 7).beta

transform(fm, Xtest)
transform(fm, Xtest; nlv = 7)

res = predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f    

res = predict(fm, Xtest; nlv = 1:2)
res.pred[1]
res.pred[2]

fm = kplsr(Xtrain, ytrain; nlv = nlv, kern = "kpol", degree = 2, 
    gamma = 1e-1, coef0 = 10) ;
res = predict(fm, Xtest)
rmsep(res.pred, ytest)

# Example of fitting the function sinc(x)
# described in Rosipal & Trejo 2001 p. 105-106 
x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
fm = kplsr(x, y; nlv = 2) ;
pred = predict(fm, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
function kplsr(X, Y, weights = ones(size(X, 1)); 
        nlv, kern = "krbf", tol = 1.5e-8, maxit = 100, kwargs...)
    kplsr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; 
        nlv = nlv, kern = kern, tol = tol, maxit = maxit, kwargs...)
end

function kplsr!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); 
        nlv, kern = "krbf", tol = 1.5e-8, maxit = 100, kwargs...)
    n = nro(X)
    q = nco(Y)
    weights = mweight(weights)
    ymeans = colmean(Y, weights)   
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
    (beta = beta, int = int)
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
    pred = list(le_nlv, Matrix{Float64})
    @inbounds for i = 1:le_nlv
        z = coef(object; nlv = nlv[i])
        pred[i] = z.int .+ @view(T[:, 1:nlv[i]]) * z.beta
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end
