"""
    center()
    center(X)
    center(X, weights::Weight)
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

function center(X)
    xmeans = colmean(X)
    Center(xmeans)
end

function center(X, weights::Weight)
    xmeans = colmean(X, weights)
    Center(xmeans)
end

""" 
    transf(object::Center, X)
    transf!(object::Center, X::Matrix)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Center, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Center, X::Matrix)
    fcenter!(X, object.xmeans)
end

"""
    scale()
    scale(X)
    scale(X, weights::Weight)
Column-wise scaling of X-data.
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

model = scale() 
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

function scale(X)
    xscales = colstd(X)
    Scale(xscales)
end

function scale(X, weights::Weight)
    xscales = colstd(X, weights)
    Scale(xscales)
end

""" 
    transf(object::Scale, X)
    transf!(object::Scale, X::Matrix)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Scale, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Scale, X::Matrix)
    fscale!(X, object.xscales)
end

"""
    cscale()
    cscale(X)
    cscale(X, weights::Weight)
Column-wise centering and scaling of X-data.
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

model = cscale() 
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

function cscale(X)
    xmeans = colmean(X)
    xscales = colstd(X)
    Cscale(xmeans, xscales)
end

function cscale(X, weights::Weight)
    xmeans = colmean(X, weights)
    xscales = colstd(X, weights)
    Cscale(xmeans, xscales)
end

""" 
    transf(object::Cscale, X)
    transf!(object::Cscale, X::Matrix)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Cscale, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Cscale, X::Matrix)
    fcscale!(X, object.xmeans, object.xscales)
end

