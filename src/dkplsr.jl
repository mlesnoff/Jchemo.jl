"""
    dkplsr(; kwargs...)
    dkplsr(X, Y; kwargs...)
    dkplsr(X, Y, weights::Weight; kwargs...)
    dkplsr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Direct kernel partial least squares regression (DKPLSR) (Bennett & Embrechts 2003).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to consider. 
* `kern` : Type of kernel used to compute the Gram matrices. Possible values are: `:krbf`, `:kpol`. See respective functions 
    `krbf` and `kpol` for their keyword arguments.
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected standard deviation.

The method builds kernel Gram matrices and then runs a usual PLSR algorithm on them. This is faster (but not equivalent) to the 
"true" KPLSR (Nipals) algorithm (function `kplsr`) described in Rosipal & Trejo (2001).

## References 
Bennett, K.P., Embrechts, M.J., 2003. An optimization perspective on kernel partial least squares regression, 
in: Advances in Learning Theory: Methods, Models and Applications, NATO Science Series III: Computer & Systems Sciences. 
IOS Press Amsterdam, pp. 227-250.

Rosipal, R., Trejo, L.J., 2001. Kernel Partial Least Squares Regression in Reproducing Kernel Hilbert Space. 
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
kern = :krbf ; gamma = 1e-1 ; scal = false
#gamma = 1e-4 ; scal = true
model = dkplsr(; nlv, kern, gamma, scal) ;
fit!(model, Xtrain, ytrain)
@names model
fitm = model.fitm ;
typeof(fitm)
@names fitm
typeof(fitm.fitm)
@names fitm.fitm

@head transf(model, Xtrain)
@head fitm.fitm.T

@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

coef(model)
coef(model; nlv = 3)

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
gamma = 1 / 3
model = dkplsr(; nlv, gamma) ;
fit!(model, x, y)
pred = predict(model, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
dkplsr(; kwargs...) = JchemoModel(dkplsr, nothing, kwargs)

function dkplsr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    dkplsr(X, Y, weights; kwargs...)
end

function dkplsr(X, Y, weights::Weight; kwargs...)
    dkplsr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function dkplsr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
    par = recovkw(ParKplsr, kwargs).par
    @assert in([:krbf ; :kpol])(par.kern) "Wrong value for argument 'kern'." 
    Q = eltype(X)
    isa(Y, BitMatrix) ? Y = convert.(Q, Y) : nothing
    p = nco(X)
    q = nco(Y)
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        fscale!(X, xscales)
        fscale!(Y, yscales)
    end
    fkern = eval(Meta.parse(string("Jchemo.", par.kern)))
    K = fkern(X, X; kwargs...)     
    fitm = plskern!(K, Y, weights; kwargs...)
    Dkplsr(fitm, X, K, xscales, yscales, kwargs, par) 
end

""" 
    transf(object::Dkplsr, X; nlv = nothing)
Compute latent variables (LVs; = scores) from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Dkplsr, X; nlv = nothing)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(fscale(X, object.xscales), object.X; values(object.kwargs)...)
    transf(object.fitm, K; nlv)
end

"""
    coef(object::Dkplsr; nlv = nothing)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function coef(object::Dkplsr; nlv = nothing)
    coef(object.fitm; nlv)
end

"""
    predict(object::Dkplsr, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
   
""" 
function predict(object::Dkplsr, X; nlv = nothing)
    a = nco(object.fitm.T)
    isnothing(nlv) ? nlv = a : nlv = (min(a, minimum(nlv)):min(a, maximum(nlv)))
    le_nlv = length(nlv)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(fscale(X, object.xscales), object.X; object.kwargs...)
    pred = predict(object.fitm, K; nlv).pred
    if le_nlv == 1
        pred .= pred * Diagonal(object.yscales)
    else
        for i = 1:le_nlv
            pred[i] .= pred[i] * Diagonal(object.yscales)
        end
    end
    (pred = pred,)
end
