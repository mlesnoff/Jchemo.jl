"""
    center()
    center(X; kwargs...)
    center(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...)
Column-wise centering of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `typ` : Symbol defining the column centering. Possible values are: `:none` (no centering), 
    and `:mean` (default). 

# Examples
```julia
using Jchemo 
X = [1. 2 3 4; 4 1 6 7; 12 5 6 13; 27 18 7 6; 12 11 28 7] 
Xnew = X[1:3, :] .+ collect(1:3)
n = nro(X)
w = collect(1:n)
weights = pweight(w)

model = center() 
fit!(model, X)
@names fitm = model.fitm
hcat(fitm.centers, colmean(X))
Xp = transf(model, X)
X .- colmean(X)'
Xpnew = transf(model, Xnew)
Xnew .- colmean(X)'

model = center() 
fit!(model, X, weights)
@names fitm = model.fitm
hcat(fitm.centers, colmean(X, weights))
Xp = transf(model, X)
X .- colmean(X, weights)'
Xpnew = transf(model, Xnew)
Xnew .- colmean(X, weights)'

center(X; typ = :none).centers
hcat(center(X; typ = :mean).centers, colmean(X))

center(X, weights; typ = :none).centers
hcat(center(X, weights; typ = :mean).centers, colmean(X, weights))
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
    @assert @in(par.typ, [:none, :mean]) "Wrong value for argument 'centr'."
    if par.typ == :none
        v = zeros(Q, nco(X))
    elseif par.typ == :mean 
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
* `typ` : Symbol defining the column scaling. Possible values are: `:none` (no scaling), 
    `:std` (uncorrected STD; default), `:prt` (Pareto), and `:mad` (MAD).

# Examples
```julia 
using Jchemo 
X = [1. 2 3 4; 4 1 6 7; 12 5 6 13; 27 18 7 6; 12 11 28 7] 
Xnew = X[1:3, :] .+ collect(1:3)
n = nro(X)
w = collect(1:n)
weights = pweight(w)

model = scale() 
fit!(model, X)
@names fitm = model.fitm
hcat(fitm.scales, colstd(X))
Xp = transf(model, X)
X * diagm(1 ./ colstd(X)) 
Xpnew = transf(model, Xnew)
Xnew * diagm(1 ./ colstd(X))

model = scale() 
fit!(model, X, weights)
@names fitm = model.fitm
hcat(fitm.scales, colstd(X, weights))
Xp = transf(model, X)
X * diagm(1 ./ colstd(X, weights)) 
Xpnew = transf(model, Xnew)
Xnew * diagm(1 ./ colstd(X, weights))

scale(X; typ = :none).scales
hcat(scale(X; typ = :std).scales, colstd(X))

scale(X, weights; typ = :none).scales
hcat(scale(X, weights; typ = :std).scales, colstd(X, weights))
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
    @assert @in(par.typ, [:none, :std, :prt, :mad]) "Wrong value for argument 'scal'."
    if par.typ == :none
        v = ones(Q, nco(X))
    elseif par.typ == :std
        v = colstd(X, weights)
    elseif par.typ == :prt
        v = colprt(X, weights)
    elseif par.typ == :mad
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
* `centr` : Symbol defining the column centering. Possible values are: `:none` (no centering), 
    and `:mean` (default). 
* `scal` : Symbol defining the column scaling. Possible values are: `:none` (no scaling), 
    `:std` (uncorrected STD; default), `:prt` (Pareto), and `:mad` (MAD).

# Examples
```julia
using Jchemo 
X = [1. 2 3 4; 4 1 6 7; 12 5 6 13; 27 18 7 6; 12 11 28 7] 
Xnew = X[1:3, :] .+ collect(1:3)
n = nro(X)
w = collect(1:n)
weights = pweight(w)

centr = :mean ; scal = :std
#centr = :mean ; scal = :none
#centr = :none ; scal = :std
#centr = :none ; scal = :none
model = cscale(; centr, scal) 
fit!(model, X)
@names fitm = model.fitm 
hcat(fitm.centers, colmean(X))
hcat(fitm.scales, colstd(X))
Xp = transf(fitm, X)
(X .- colmean(X)') * diagm(1 ./ colstd(X)) 
Xpnew = transf(model, Xnew)
(Xnew .- colmean(X)') * diagm(1 ./ colstd(X)) 

centr = :mean ; scal = :std
#centr = :mean ; scal = :none
#centr = :none ; scal = :std
#centr = :none ; scal = :none
model = cscale(; centr, scal) 
fit!(model, X, weights)
@names fitm = model.fitm 
hcat(fitm.centers, colmean(X, weights))
hcat(fitm.scales, colstd(X, weights))
Xp = transf(fitm, X)
(X .- colmean(X, weights)') * diagm(1 ./ colstd(X, weights)) 
Xpnew = transf(model, Xnew)
(Xnew .- colmean(X, weights)') * diagm(1 ./ colstd(X, weights)) 
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
    centers = center(X, weights; typ = par.centr).centers
    scales = scale(X, weights; typ = par.scal).scales
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

