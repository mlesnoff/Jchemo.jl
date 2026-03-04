## Approach KNN-prototypes:
## - Some observations are sampled within Train, and represent "prototypes".
## - A neighborhood is selected around each prototype. Note: A same training observation 
##   can belong to several neighborhoods.
## - A PLSR (prototype) model is optimized on each neighborhood, using a K-fold cross-validation,
##   and stored.
## - Each new observation to predict is assigned to its k nearest prototypes, and the final 
##   prediction is obtained by a weighted average of the k predictions of the corresponding 
##   prototype models.

## The present function implements this approach only for univariate Y.

## --- Parameters to define the prototype models
## * n_proto : Total nb. prototypes
## * nlvdis : Nb. latent variables for the dissimilarity model (0: Euclidean) 
## * metric :  Metric for the dissimilarity (e.g. :eucl, :mah, :cor)
## * k_proto : Nb. neighbors for each prototype
## * nlv_proto : Max. nb. latent variables for each prototype model  
## * scal_proto : Scale the (local) data before fitting each prototype model
## --- Parameters for final prediction (weighted average of the prototype predictions)
## * [The metric is the same as the one used to select the prototype samples]
## * k : Nb. prototype models whose prédictions are averaged
## * h : Sharpeness of the weighting function used when averaging
## * criw : See function `winvs`
## * squared : Whether squared distances are used or not to compute the weights
## * tolw : Tolerance (stabilizes predictions when many weights are close to zero; same as in lwplsr)

struct ProtoPlsr
    Xproto::Matrix
    Yproto::Matrix
    fitm_emb::Union{Nothing, Plsr}
    resnn::Vector
    fitm
    coefs::Vector
    par::ParProtoPlsr
end

struct ParProtoPlsr
    n_proto::Int = 1
    nlvdis::Int = 0                         
    metric::Symbol = :eucl  
    k_proto::Int = 1 
    centroid_proto::Int = true
    nlv_proto::Union{Int, Vector{Int}, UnitRange} = 1   
    scal_proto::Bool = false    
    k::Int = 1                              
    h::Float64 = Inf                        
    criw::Float64 = 4                       
    squared::Bool = false                   
    tolw::Float64 = 1e-4                    
end

function protoplsr(X, Y; kwargs...)
    par = recovkw(ParProtoPlsr, kwargs).par 
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = nro(X)
    ## Selection of the prototypes
    #s_proto = sort(sampks(X; k = n_proto, metric = "eucl").train)
    s_proto = samprand(n, par.n_proto; seed = 1234).test
    ## Compute the neighborhood of each prototype
    if nlvdis == 0
        fitm_emb = nothing
        resnn = getknn(X, vrow(X, s_proto); k = par.k_proto, par.metric)
    else
        fitm_emb = plskern(X, Y; nlv = par.nlvdis)
        resnn = getknn(fitm_emb.T, transform(fitm_emb, vrow(X, s_proto)); k = par.k_proto, par.metric)
    end
    ## Optimize/fit the prototype models 
    fitm = list(Jchemo.Plsr, par.n_proto)
    coefs = list(NamedTuple, par.n_proto)
    segm = segmkf(k_proto, 3; rep = 1, seed = 1234)
    #@inbounds for i in eachindex(s_proto) 
    Xproto = similar(X, par.n_proto, nco(X))
    Yproto = similar(Y, par.n_proto, nco(Y))
    Threads.@threads for i in eachindex(s_proto)
        vX = vrow(X, resnn.ind[i])
        vY = vrow(Y, resnn.ind[i])
        pars = mpar(scal = par.scal_proto)
        model = plskern()
        rescv = gridcv(model, vX, vY; segm, score = rmsep, pars, nlv = 0:par.nlv_proto).res
        u = findall(rescv.y1 .== minimum(rescv.y1))[1]
        fitm[i] = plskern(vX, vY; nlv = rescv.nlv[u], scal = rescv.scal[u])
        coefs[i] = coef(fitm[i])
        if par.centroid_proto
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
    n_proto = nro(object.Xproto)
    metric = object.pars.metric
    k = min(object.pars.k, n_proto)  # nb prototype models averaged
    h = object.pars.h
    criw = object.pars.criw
    squared = object.pars.squared
    tolw = object.pars.tolw
    # Compute the neighborhood (within prototypes) of each new observation
    if isnothing(object.fitm_emb)
        res = getknn(object.Xproto, X; k, metric)
    else
        res = getknn(object.fitm_emb.T, transform(object.fitm_emb, X); k, metric) 
    end
    listnn = res.ind
    listw = list(Vector{Q}, m)
    pred = zeros(m, q)
    @inbounds for i = 1:m
    #Threads.@threads for i = 1:m   # seems to not give better performance
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



