## --- Parameters to define the prototype models
## * nproto : Total nb. prototypes
## * nlvdis : Nb. latent variables for the dissimilarity model (0: Euclidean) 
## * metric :  Metric for the dissimilarity (e.g. :eucl, :mah, :cor)
## * k : Nb. neighbors for each prototype
## * nlv : Max. nb. latent variables for each prototype model  
## * scal : Scale the (local) data before fitting each prototype model
## --- Parameters for final prediction (weighted average of the prototype predictions)
## * [The metric is the same as the one used to select the prototype samples]
## * k : Nb. prototype models whose prédictions are averaged
## * h : Sharpeness of the weighting function used when averaging
## * criw : See function `winvs`
## * squared : Whether squared distances are used or not to compute the weights
## * tolw : Tolerance (stabilizes predictions when many weights are close to zero; same as in protoplsr)
"""
    protoplsr(; kwargs...)
    protoplsr(X, Y; kwargs...)
k-Nearest-Neighbours locally weighted partial least squares regression (kNN-protoplsr).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
Keyword arguments:
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS used for the dimension 
    reduction before computing the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors and to compute the weights 
    (see function `getknn`). Possible values are: `:eucl` (Euclidean), `:mah` (Mahalanobis), 
    `:sam` (spectral angular distance), `:cos` (cosine distance), `:cor` (correlation distance).
* `nproto`: Nb. prototype models.
* `k`: Nb. observations considered in each 
* `centroid`:
* `samp`:
* `nlv` : Nb. latent variables (LVs) for the local (i.e. inside each neighborhood) models.
* `kavg` : The number of nearest neighbors to select for each observation to predict.
* `h` : A scalar defining the shape of the weight function computed by function `winvs`. Lower is h, 
    sharper is the function. See function `winvs` for details (keyword arguments `criw` and `squared` of 
    `winvs` can also be specified here).
* `tolw` : For stabilization when very close neighbors.
* `scal` : Boolean. If `true`, (a) each column of the global `X` (and of the global `Y` if there 
    is a preliminary PLS reduction dimension) is scaled by its uncorrected standard deviation before to compute 
    the distances and the weights, and (b) the X and Y scaling is also done within each neighborhood (local level) 
    for the weighted PLSR.

Function `protoplsr` fits a KNN prototype model approach:
* Some observations are sampled within Train, and represent "prototypes".
* A neighborhood is selected around each prototype. Note: A same training observation 
  can belong to several neighborhoods.
* A PLSR (prototype) model is optimized on each neighborhood, using a K-fold cross-validation,
   and stored.
* Each new observation to predict is assigned to its k nearest prototypes, and the final 
  prediction is obtained by a weighted average of the k predictions of the corresponding 
  prototype models.

The present function implements this approach only for univariate Y.



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

nlvdis = 15 ; metric = :mah 
k = 500 ; h = 1 ; nlv = 10
model = protoplsr(; nlvdis, metric, h, k, nlv) 
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm

res = predict(model, Xtest) ; 
@names res 
res.listnn
res.listd
res.listw
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",  
    ylabel = "Observed").f    
```
""" 
protoplsr(; kwargs...) = JchemoModel(protoplsr, nothing, kwargs)

Base.@kwdef mutable struct ParProtoPlsr
    nlvdis::Int = 0                         
    metric::Symbol = :eucl  
    nproto::Int = 1
    k::Int = 1 
    centroid::Bool = true
    samp::Symbol = :rand
    nlv::Union{Int, Vector{Int}, UnitRange} = 1   
    kavg::Int = 1                              
    h::Float64 = Inf                        
    criw::Float64 = 4                       
    squared::Bool = false                   
    tolw::Float64 = 1e-4                    
    scal::Bool = false    
end

struct ProtoPlsr
    Xproto::Matrix
    Yproto::Matrix
    fitm_emb::Union{Nothing, Plsr}
    resnn::NamedTuple
    fitm
    coefs::Vector
    par::ParProtoPlsr
end

function protoplsr(X, Y; kwargs...)
    par = recovkw(ParProtoPlsr, kwargs).par 
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = nro(X)
    ## Selection of the prototypes
    if par.samp == :rand
        s_proto = samprand(n, par.nproto; seed = 1234).test
    elseif par.samp == :ks
        s_proto = sampks(X, par.nproto; metric).test
    end
    ## Compute the neighborhood of each prototype
    if par.nlvdis == 0
        fitm_emb = nothing
        resnn = getknn(X, vrow(X, s_proto); k = par.k, par.metric)
    else
        fitm_emb = plskern(X, Y; nlv = par.nlvdis)
        resnn = getknn(fitm_emb.T, transform(fitm_emb, vrow(X, s_proto)); k = par.k, par.metric)
    end
    ## Optimize/fit the prototype models 
    fitm = list(Plsr, par.nproto)
    coefs = list(NamedTuple, par.nproto)
    segm = segmkf(par.k, 3; rep = 1, seed = 1234)
    Xproto = similar(X, par.nproto, nco(X))
    Yproto = similar(Y, par.nproto, nco(Y))
    @inbounds for i in eachindex(s_proto) 
    #Threads.@threads for i in eachindex(s_proto)
        vX = vrow(X, resnn.ind[i])
        vY = vrow(Y, resnn.ind[i])
        pars = mpar(scal = par.scal)
        model = plskern()
        rescv = gridcv(model, vX, vY; segm, score = rmsep, pars, nlv = 0:par.nlv).res
        u = findall(rescv.y1 .== minimum(rescv.y1))[1]
        fitm[i] = plskern(vX, vY; nlv = rescv.nlv[u], scal = rescv.scal[u])
        coefs[i] = coef(fitm[i])
        if par.centroid          
            Xproto[i, :] .= colmean(vX)
            Yproto[i, :] .= colmean(vY)    
        else
            Xproto[i, :] .= vrow(X, s_proto[i])
            Yproto[i, :] .= vrow(Y, s_proto[i])
        end
    end
    ## Outputs
    ProtoPlsr(Xproto, Yproto, fitm_emb, resnn, fitm, coefs, par) 
end

function predict(object::ProtoPlsr, X)
    X = ensure_mat(X)
    Q = eltype(X)
    m = nro(X)
    q = nco(object.Yproto)
    nproto = object.par.nproto
    metric = object.par.metric
    kavg = min(object.par.kavg, nproto)  # nb prototype models averaged
    h = object.par.h
    criw = object.par.criw
    squared = object.par.squared
    tolw = object.par.tolw
    # Compute the neighborhood (within prototypes) of each new observation
    if isnothing(object.fitm_emb)
        res = getknn(object.Xproto, X; k = kavg, metric)
    else
        res = getknn(object.fitm_emb.T, transform(object.fitm_emb, X); k = kavg, metric) 
    end
    listnn = res.ind
    listw = list(Vector{Q}, m)
    pred = zeros(m, q)
    #@inbounds for i = 1:m
    Threads.@threads for i = 1:m   # seems to not give better performance
        s = listnn[i]
        w = winvs(res.d[i]; h, criw, squared)
        w[w .< tolw] .= tolw  # same as in protoplsr
        w ./= sum(w)
        @inbounds for j in eachindex(s)
            coefs = object.coefs[s[j]]
            zpred = vec(coefs.int .+ vrow(X, i:i) * coefs.B)
            pred[i, :] .+= w[j] * zpred
        end
        listw[i] = w 
    end
    (pred = pred, listnn, listd = res.d, listw = listw)
end



