"""
    center()
    center(X; kwargs...)
    center(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...)
Column-wise centering of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `centr` : Symbol defining the X centering. Possible values are: `:none` (no centering), 
    and `:std` (uncorrected STD; default). 

# Examples
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

function center(X; kwargs...)
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    center(X, weights; kwargs...)
end 
    
function center(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    par = Jchemo.recovkw(ParCenter, kwargs).par
    @assert @in(par.centr, [:none, :mean]) "Wrong value for argument 'centr'."
    if par.centr == :none
        v = zeros(Q, nco(X))
    elseif par.centr == :mean 
         v = colmean(X, weights)
    end
    Center(v, par)
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
    fcenter!(X, object.centers)
end

"""
    scale()
    scale(X; kwargs...)
    scale(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...)
Column-wise scaling of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `scal` : Symbol defining the X scaling. Possible values are: `:none` (default), `:std` (uncorrected STD), 
    `:prt` (Pareto) and `:mad` (MAD).

# Examples
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
    
function scale(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
    par = Jchemo.recovkw(ParScale, kwargs).par
    @assert @in(par.scal, [:none, :std, :prt, :mad]) "Wrong value for argument 'scal'."
    if par.scal == :none
        v = ones(Q, nco(X))
    elseif par.scal == :std
        v = colstd(X, weights)
    elseif par.scal == :prt
        v = colprt(X, weights)
    elseif par.scal == :mad
        v = colmad(X, weights)
    end
    Scale(v, par)
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
    fscale!(X, object.scales)
end

"""
    cscale()
    cscale(X, weights; kwargs...)
    cscale(X::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
Column-wise centering and scaling of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `centr` : Symbol defining the X centering. Possible values are: `:none` (no centering), 
    and `:std` (uncorrected STD; default). 
* `scal` : Symbol defining the scaling. Possible values are: `:none` (default), `:std` (uncorrected STD), 
    `:prt` (Pareto) and `:mad` (MAD).

# Examples
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
    par = Jchemo.recovkw(ParCscale, kwargs).par 
    println(par)
    centers = center(X, weights; kwargs...).centers
    scales = scale(X, weights; kwargs...).scales
    Cscale(centers, scales, par)
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
    fcscale!(X, object.centers, object.scales)
end

