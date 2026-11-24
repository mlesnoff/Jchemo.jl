"""
    center()
    center(X; kwargs...)
    center(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...)
Column-wise centering of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `centrx` : Symbol defining the column centering. Possible values are: `:none` (no centering), 
    `:mean` (default), and `:med` (median). 

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

center(X; centrx = :none).centers
hcat(center(X; centrx = :mean).centers, colmean(X))

center(X, weights; centrx = :none).centers
hcat(center(X, weights; centrx = :mean).centers, colmean(X, weights))
```
"""
center(; kwargs...) = JchemoModel(center, nothing, kwargs)

function center(X; kwargs...)
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    center(X, weights; kwargs...)
end 
    
function center(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
    par = Jchemo.recovkw(ParCenter, kwargs).par
    @assert @in(par.centrx, [:none, :mean, :med]) "Wrong value for argument 'centrx'."
    if par.centrx == :none
        v = zeros(Q, nco(X))
    elseif par.centrx == :mean 
         v = colmean(X, weights)
    elseif par.centrx == :med 
        v = colmed(X)   # no weighting for medians
    end
    Center(v, par)
end

""" 
    transf(object::Center, X)
    transf!(object::Center, X::AbstractMatrix{Q}) where {Q<:Float}
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Center, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Center, X::AbstractMatrix{Q}) where {Q<:Float}
    if !(object.par.centrx == :none)
        fcenter!(X, object.centers)
    end
end

"""
    scale()
    scale(X; kwargs...)
    scale(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...)
Column-wise scaling of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `scalx` : Symbol defining the column scaling. Possible values are: `:none` (no scaling), 
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

scale(X; scalx = :none).scales
hcat(scale(X; scalx = :std).scales, colstd(X))

scale(X, weights; scalx = :none).scales
hcat(scale(X, weights; scalx = :std).scales, colstd(X, weights))
```
"""
scale(; kwargs...) = JchemoModel(scale, nothing, kwargs)

function scale(X; kwargs...)
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    scale(X, weights; kwargs...)
end 
    
function scale(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
    par = Jchemo.recovkw(ParScale, kwargs).par
    @assert @in(par.scalx, [:none, :std, :prt, :mad]) "Wrong value for argument 'scalx'."
    if par.scalx == :none
        v = ones(Q, nco(X))
    elseif par.scalx == :std
        v = colstd(X, weights)
    elseif par.scalx == :prt
        v = colprt(X, weights)
    elseif par.scalx == :mad
        v = colmad(X, weights)
    end
    Scale(v, par)
end

""" 
    transf(object::Scale, X)
    transf!(object::Scale, X::AbstractMatrix{Q}) where {Q<:Float}
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Scale, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Scale, X::AbstractMatrix{Q}) where {Q<:Float}
    if !(object.par.scalx == :none)
        fscale!(X, object.scales)
    end
end

"""
    cscale()
    cscale(X, weights; kwargs...)
    cscale(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
Column-wise centering and scaling of X-data.
* `X` : X-data (n, p).
Keyword arguments:
* `centrx` : Symbol defining the column centering. Possible values are: `:none` (no centering), 
    `:mean` (default), and `:med` (median). 
* `scalx` : Symbol defining the column scaling. Possible values are: `:none` (no scaling), 
    `:std` (uncorrected STD; default), `:prt` (Pareto), and `:mad` (MAD).

# Examples
```julia
using Jchemo 
X = [1. 2 3 4; 4 1 6 7; 12 5 6 13; 27 18 7 6; 12 11 28 7] 
Xnew = X[1:3, :] .+ collect(1:3)
n = nro(X)
w = collect(1:n)
weights = pweight(w)

centrx = :mean ; scalx = :std
#centrx = :mean ; scalx = :none
#centrx = :none ; scalx = :std
#centrx = :none ; scalx = :none
model = cscale(; centrx, scalx) 
fit!(model, X)
@names fitm = model.fitm 
hcat(fitm.centers, colmean(X))
hcat(fitm.scales, colstd(X))
Xp = transf(fitm, X)
(X .- colmean(X)') * diagm(1 ./ colstd(X)) 
Xpnew = transf(model, Xnew)
(Xnew .- colmean(X)') * diagm(1 ./ colstd(X)) 

centrx = :mean ; scalx = :std
#centrx = :mean ; scalx = :none
#centrx = :none ; scalx = :std
#centrx = :none ; scalx = :none
model = cscale(; centrx, scalx) 
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

function cscale(X::AbstractMatrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
    par = Jchemo.recovkw(ParCscale, kwargs).par 
    centers = center(X, weights; centrx = par.centrx).centers
    scales = scale(X, weights; scalx = par.scalx).scales
    Cscale(centers, scales, par)
end

""" 
    transf(object::Cscale, X)
    transf!(object::Cscale, X::AbstractMatrix{Q}) where {Q<:Float}
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Cscale, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Cscale, X::AbstractMatrix{Q}) where {Q<:Float}
    if !(object.par.centrx == :none)
        fcenter!(X, object.centers)
    end
    if !(object.par.scalx == :none)
        fscale!(X, object.scales)
    end
end

