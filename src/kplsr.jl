"""
    kplsr(; kwargs...)
    kplsr(X, Y; kwargs...)
    kplsr(X, Y, weights::Weight; kwargs...)
    kplsr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Kernel partial least squares regression (KPLSR) implemented with a Nipals 
    algorithm (Rosipal & Trejo, 2001).
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
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
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
model = kplsr(; nlv, kern, gamma) ;
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm
@head model.fitm.T

coef(model)
coef(model; nlv = 3)

@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
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
model = kplsr(; nlv, kern, gamma) 
fit!(model, x, y)
pred = predict(model, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
kplsr(; kwargs...) = JchemoModel(kplsr, nothing, kwargs)

function kplsr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    kplsr(X, Y, weights; kwargs...)
end

function kplsr(X, Y, weights::Weight; kwargs...)
    kplsr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function kplsr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
    par = recovkw(ParKplsr, kwargs).par
    @assert in([:krbf ; :kpol])(par.kern) "Wrong value for argument 'kern'." 
    Q = eltype(X)
    isa(Y, BitMatrix) ? Y = convert.(Q, Y) : nothing
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
    Kt = K'    
    DKt = fweight(Kt, weights.w)
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- vtot .+ sum(fweight(DKt', weights.w)) 
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
            mul!(t, K, vec(fweight(Y, weights.w)))  # t = K * D * Y
            t ./= sqrt(dot(t, weights.w .* t))
            dt .= weights.w .* t
            mul!(c, Y', dt)
            u .= Y * c 
            u ./= normv(u) 
        else
            u .= Y[:, 1]
            ztol = 1.
            ziter = 1
            while ztol > par.tol && ziter <= par.maxit
                mul!(t, K, weights.w .* u)
                t ./= normv(t, weights) 
                dt .= weights.w .* t                
                mul!(c, Y', dt)
                zu .= Y * c 
                zu ./= normv(zu) 
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
    DU = fweight(U, weights.w)
    zR = DU * inv(T' * fweight(Kc, weights.w) * DU)   # = DU * inv(T' * D * Kc * DU)
    Kplsr(X, Kt, T, C, U, zR, DKt, vtot, xscales, ymeans, yscales, weights, iter, kwargs, par) 
end

""" 
    transf(object::Kplsr, X; nlv = nothing)
Compute latent variables (LVs = scores) from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Kplsr, X; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(fscale(X, object.xscales), object.X; object.kwargs...)
    DKt = fweight(K', object.weights.w) 
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- object.vtot .+ sum(fweight(object.DKt', object.weights.w))  
    Kc * @view(object.R[:, 1:nlv])
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
    (beta = beta, int)
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
    isnothing(nlv) ? nlv = a : nlv = min(a, minimum(nlv)):min(a, maximum(nlv))
    le_nlv = length(nlv)
    T = transf(object, X)
    pred = list(Matrix{eltype(X)}, le_nlv)
    @inbounds for i in eachindex(nlv)
        z = coef(object; nlv = nlv[i])
        pred[i] = z.int .+ @view(T[:, 1:nlv[i]]) * z.beta * Diagonal(object.yscales)
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end
