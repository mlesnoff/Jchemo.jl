"""
    kplsr(; kwargs...)
    kplsr(X, Y; kwargs...)
    kplsr(X, Y, weights::Weight; kwargs...)
    kplsr!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Kernel partial least squares regression (KPLSR) implemented 
    with a Nipals algorithm (Rosipal & Trejo, 2001).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to consider. 
* `kern` : Type of kernel used to compute the Gram matrices.
    Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.

This algorithm becomes slow for n > 1000. 
Use function `dkplsr` instead.

## References 
Rosipal, R., Trejo, L.J., 2001. Kernel Partial Least 
Squares Regression in Reproducing Kernel Hilbert Space. 
Journal of Machine Learning Research 2, 97-123.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 20
kern = :krbf ; gamma = 1e-1
mod = kplsr(; nlv, kern, gamma) ;
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
@head mod.fm.T

coef(mod)
coef(mod; nlv = 3)

@head transf(mod, Xtest)
@head transf(mod, Xtest; nlv = 3)

res = predict(mod, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####### Example of fitting the function sinc(x)
####### described in Rosipal & Trejo 2001 p. 105-106 
x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
nlv = 2
kern = :krbf ; gamma = 1 / 3
mod = kplsr(; nlv, kern, gamma) ;
fit!(mod, x, y)
pred = predict(mod, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
function kplsr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    kplsr(X, Y, weights; kwargs...)
end

function kplsr(X, Y, weights::Weight; kwargs...)
    kplsr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function kplsr!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert in([:krbf ; :kpol])(par.kern) "Wrong value for argument 'kern'." 
    Q = eltype(X)
    n, p = size(X)
    q = nco(Y)
    nlv = par.nlv
    ymeans = colmean(Y, weights)   
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        fscale!(X, xscales)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(Y, ymeans)
    end
    fkern = eval(Meta.parse(string("Jchemo.", par.kern)))  
    K = fkern(X, X; kwargs...)     # In the future?: fkern!(K, X, X; values(kwargs)...)
    D = Diagonal(weights.w)
    Kt = K'    
    DKt = D * Kt
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- vtot .+ sum(D * DKt')
    ## Pre-allocation
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
            t ./= sqrt(dot(t, weights.w .* t))
            dt .= weights.w .* t
            mul!(c, Y', dt)
            u .= Y * c 
            u ./= sqrt(dot(u, u))
        else
            u .= Y[:, 1]
            ztol = 1.
            ziter = 1
            while ztol > par.tol && ziter <= par.maxit
                mul!(t, K, weights.w .* u)
                t ./= sqrt(dot(t, weights.w .* t))
                dt .= weights.w .* t                
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
    Kplsr(X, Kt, T, C, U, zR, D, DKt, vtot, xscales, ymeans, yscales, 
        weights, iter, kwargs, par)
end

""" 
    transf(object::Kplsr, X; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Kplsr, X; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(fscale(X, object.xscales), object.X; 
        object.kwargs...)
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
   
""" 
function coef(object::Kplsr; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    beta = object.C[:, 1:nlv]'
    q = length(object.ymeans)
    int = reshape(object.ymeans, 1, q)
    (beta = beta, int = int)
end

"""
    predict(object::Kplsr, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
If nothing, it is the maximum nb. LVs.
""" 
function predict(object::Kplsr, X; nlv = nothing)
    X = ensure_mat(X)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    T = transf(object, X)
    pred = list(Matrix{eltype(X)}, le_nlv)
    @inbounds for i = 1:le_nlv
        z = coef(object; nlv = nlv[i])
        pred[i] = z.int .+ @view(T[:, 1:nlv[i]]) * z.beta * Diagonal(object.yscales)
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end
