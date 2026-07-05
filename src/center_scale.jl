"""
    center()
    center(X)
    center(X:: Matrix{Q}, weights::ProbabilityWeights) where Q <: Float
Column-wise centering of X-data.
* `X` : X-data (n, p).

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

model = center() 
fit!(model, Xtrain)
Xptrain = transf(model, Xtrain)
Xptest = transf(model, Xtest)
colmean(Xptrain)
@head Xptest 
@head Xtest .- colmean(Xtrain)'
plotsp(Xptrain).f
plotsp(Xptest).f
```
"""
center(; kwargs...) = JchemoModel(center, nothing, kwargs)

center(X) = Center(colmean(ensure_mat(X)))

function center(X::Matrix{Q}, weights::ProbabilityWeights{Q}) where Q <: Float
    xmeans = colmean(X, weights)
    Center(xmeans)
end

""" 
    transf(object::Center, X)
    transf!(object::Center, X::Matrix{Q}) where Q <: Float
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Center, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Center, X::Matrix{Q}) where Q <: Float
    fcenter!(X, object.xmeans)
end

"""
    scale()
    scale(X; kwargs...)
    scale(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
Column-wise scaling of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `scal` : Symbol defining the scaling. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

## Examples
```julia 
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

model = scale(scal = :std) 
fit!(model, Xtrain)
Xptrain = transf(model, Xtrain)
Xptest = transf(model, Xtest)
colstd(Xptrain)
@head Xptest 
@head Xtest ./ colstd(Xtrain)'
plotsp(Xptrain).f
plotsp(Xptest).f
```
"""
scale(; kwargs...) = JchemoModel(scale, nothing, kwargs)

function scale(X; kwargs...)
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    scale(X, weights; kwargs...)
end

function scale(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    par = Jchemo.recovkw(ParScale, kwargs).par 
    xscales = ones(Q, nco(X)) 
    if par.scal != :none
        colscal = def_colscal(par.scal) 
        xscales .= colscal(X, weights)
    end
    Scale(xscales)
end

""" 
    transf(object::Scale, X)
    transf!(object::Scale, X::Matrix{Q}) where Q <: Float
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Scale, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Scale, X::Matrix{Q}) where Q <: Float
    fscale!(X, object.xscales)
end

"""
    cscale()
    cscale(X, weights; kwargs...)
    cscale(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
Column-wise centering and scaling of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `scal` : Symbol defining the scaling. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))

db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

model = cscale(scal = :std) 
fit!(model, Xtrain)
Xptrain = transf(model, Xtrain)
Xptest = transf(model, Xtest)
colmean(Xptrain)
colstd(Xptrain)
@head Xptest 
@head (Xtest .- colmean(Xtrain)') ./ colstd(Xtrain)'
plotsp(Xptrain).f
plotsp(Xptest).f
```
"""
cscale(; kwargs...) = JchemoModel(cscale, nothing, kwargs)

function cscale(X; kwargs...)
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    cscale(X, weights; kwargs...)
end

function cscale(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    par = Jchemo.recovkw(ParScale, kwargs).par 
    p = nco(X) 
    xmeans = colmean(X, weights)
    xscales = ones(Q, p) 
    if par.scal != :none
        colscal = def_colscal(par.scal) 
        xscales .= colscal(X, weights)
    end
    Cscale(xmeans, xscales)
end

""" 
    transf(object::Cscale, X)
    transf!(object::Cscale, X::Matrix{Q}) where Q <: Float
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Cscale, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Cscale, X::Matrix{Q}) where Q <: Float
    fcscale!(X, object.xmeans, object.xscales)
end

