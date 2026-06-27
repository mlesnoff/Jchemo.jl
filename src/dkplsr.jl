"""
    dkplsr(; kwargs...)
    dkplsr(X, Y; kwargs...)
    dkplsr(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    dkplsr!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
Direct kernel partial least squares regression (DKPLSR) (Bennett & Embrechts 2003).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to consider. 
* `kern` : Type of kernel used to compute the Gram matrices. Possible values are: `:krbf`, `:kpol`. See respective functions 
    `krbf` and `kpol` for their keyword arguments.
* `scal` : Symbol defining the column scaling of `X` and `Y`. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

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
kern = :krbf ; gamma = 1e-1 ; scal = :none
#gamma = 1e-4 ; scal = :std
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
@head transf(model, Xtest, 3)

coef(model)
coef(model, 3)

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
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    dkplsr(X, Y, weights; kwargs...)
end

function dkplsr(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    dkplsr!(copy(X), copy(Y), weights; kwargs...)
end

function dkplsr!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    par = recovkw(ParKplsr{Q}, kwargs).par
    @assert in([:krbf ; :kpol])(par.kern) "Wrong value for argument 'kern'." 
    p = nco(X)
    q = nco(Y)
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal != :none
        colscal = def_colscal(par.scal) 
        xscales .= colscal(X, weights)
        yscales .= colscal(Y, weights)
        fscale!(X, xscales)
        fscale!(Y, yscales)
    end
    fkern = eval(Meta.parse(string("Jchemo.", par.kern)))
    K = fkern(X, X; kwargs...)     
    fitm = plskern!(K, Y, weights; kwargs...)
    Dkplsr(fitm, X, K, xscales, yscales, kwargs, par) 
end

""" 
    transf(object::Dkplsr, X)
    transf(object::Dkplsr, X, nlv::Int)
Compute latent variables (LVs; = scores) from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Dkplsr, X)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(fscale(X, object.xscales), object.X; values(object.kwargs)...)
    transf(object.fitm, K)
end

function transf(object::Dkplsr, X, nlv::Int)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(fscale(X, object.xscales), object.X; values(object.kwargs)...)
    transf(object.fitm, K, nlv)
end

"""
    coef(object::Dkplsr) = coef(object.fitm)
    coef(object::Dkplsr, nlv::Union{Nothing, Int})
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `nlv` : Nb. LVs to consider. 
""" 
coef(object::Dkplsr) = coef(object.fitm)

coef(object::Dkplsr, nlv::Union{Nothing, Int}) = coef(object.fitm, nlv)

"""
    predict(object::Dkplsr, X)
    predict(object::Dkplsr, X, nlv::Union{Int, AbstractVector{Int}})
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Dkplsr, X)
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(fscale(X, object.xscales), object.X; object.kwargs...)
    pred = predict(object.fitm, K).pred * Diagonal(object.yscales)
    (pred = pred, nlv = object.par.nlv) 
end

function predict(object::Dkplsr, X, nlv::Union{Int, AbstractVector{Int}})
    a = object.par.nlv
    if isa(nlv, Int)
        nlv = min(nlv, a)
    else
        nlv = min(minimum(nlv), a):min(maximum(nlv), a)
    end
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(fscale(X, object.xscales), object.X; object.kwargs...)
    pred = predict(object.fitm, K, nlv).pred
    D = Diagonal(object.yscales)
    @inbounds for i in eachindex(nlv)
            pred[i] .= pred[i] * D
    end
    (pred = pred, nlv)
end

