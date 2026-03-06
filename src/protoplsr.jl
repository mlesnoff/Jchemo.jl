"""
    protoplsr(; kwargs...)
    protoplsr(X, Y; kwargs...)
Averaging PLSR models built on the neighborhood of prototype observations.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
Keyword arguments:
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS used for the dimension 
    reduction before computing the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors and eventually to compute 
    the weights when averaging the `kavg` prototype models. Possible values are (see function `getknn`): 
    `:eucl` (Euclidean), `:mah` (Mahalanobis), `:sam` (spectral angular distance), `:cos` (cosine distance), `
    :cor` (correlation distance).
* `nproto`: Nb. prototypes to select.
* `k`: Nb. observations considered for each prototype (= neighborhood prototype = local neighborhood).
* `centroid`: Boolean. If `true`, the center of the prototype is defined by the mean of the neighborhood,
    else it is defined directely by the sampled observation. 
* `samp`: Type of sampling used in `X` to select the prototypes. 
* `nlv` : Maximum nb. latent variables (LVs) for each prototype model.
* `K` : Nb. folds (segments) in the K-fold cross-validation. 
* `kavg` : Nb. prototype models whose predictions are averaged to compute the final prediction.
* `h` : A scalar defining the shape of the weight function computed by function `winvs`. Lower is h, 
    sharper is the function. See function `winvs` for details (keyword arguments `criw` and `squared` of 
    `winvs` can also be specified here). Used when averaging the prototype predictions.
* `scal` : Boolean. If `true`, each column of matrices X and Y of the prototype neighborhood is 
    scaled by its uncorrected standard deviation.

Function `protoplsr` implements an averaging of prototype PLSR models.

*Model fitting*
* A number of `nproto` observations (x, y), referred to as 'prototypes', are sampled in the training data. 
    In the actual version of the function, the sampling is done on `X` or on global PLS scores computed 
    from (`X`, `Y`). 
* A neighborhood is selected around each prototype. The prototype neighborhoods are assumed to represent 
    the data heterogeneity (diversity of application domains). Note: A same observation can eventually belong 
    to several neighborhoods.
* On each class, a PLSR is optimized using a K-fold cross-validation and stored. This defines the 
    prototype model.

*Prediction*
* Each new observation to predict is assigned to its `kavg` nearest prototypes, based on its distances 
    to the prototype centers (prototypes or neighborhodd centroids).
* The final prediction is computed by a weighted average of the `kavg` predictions of the corresponding 
    prototype models. The weighting is computed from the relative distances between the new observation and 
    the `kavg` prototype centers (function `winvs`). If `kavg = 1`, only one PLSR model is used (the closest
    prototype) and there is no averaging.

Notes: 
* This pipeline is still under construction, some details could change in the future.
* The actual version of the function works for multivariate `Y` but the PLSR optimizations
    are done only based on the first Y column (this will be fixed later). 

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
nproto = 100
k = 80
nlv = 15
kavg = 5 ; h = 1 
model = protoplsr(; nlvdis, metric, nproto, k, nlv, kavg, h) 
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm

res = predict(model, Xtest) ; 
@names res 
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
    K::Int = 5    
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
    fitm::Vector{Plsr}
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
        ## To do: Add argument 'seed = nothing' 
        ## * `seed` : Optional seed (integer number) for the sampling        
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
        resnn = getknn(fitm_emb.T, transf(fitm_emb, vrow(X, s_proto)); k = par.k, par.metric)
    end
    ## Optimize/fit the prototype models 
    fitm = list(Plsr, par.nproto)
    coefs = list(NamedTuple, par.nproto)
    segm = segmkf(par.k, par.K; rep = 1, seed = 1234)
    Xproto = similar(X, par.nproto, nco(X))
    Yproto = similar(Y, par.nproto, nco(Y))
    #@inbounds for i in eachindex(s_proto) 
    Threads.@threads for i in eachindex(s_proto)
        vX = vrow(X, resnn.ind[i])
        vY = vrow(Y, resnn.ind[i])
        pars = mpar(scal = par.scal)
        model = plskern()
        rescv = gridcv(model, vX, vY; segm, score = rmsep, pars, nlv = 0:par.nlv).res
        ## To do: adapt for multivariate Y
        u = findall(rescv.y1 .== minimum(rescv.y1))[1]
        ## End
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
        T = transf(object.fitm_emb, object.Xproto)
        res = getknn(T, transf(object.fitm_emb, X); k = kavg, metric) 
    end
    listnn = res.ind
    listw = list(Vector{Q}, m)
    pred = zeros(m, q)
    #@inbounds for i = 1:m
    Threads.@threads for i = 1:m   
        s = listnn[i]
        w = winvs(res.d[i]; h, criw, squared)
        w[w .< tolw] .= tolw  # same as in lwplsr
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



