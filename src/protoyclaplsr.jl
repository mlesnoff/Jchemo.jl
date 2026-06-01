## Not exported

struct Protoyclaplsr
    fitm::Vector{Plsr}    
    fitm_emb::Union{Nothing, Plsr}
    Xproto::Matrix
    Yproto::Matrix
    coefs::Vector
    ni::Vector{Int}
    lev::Vector
    par::NamedTuple
end

function protoyclaplsr(X, Y, ycla; nlvdis = 0, metric = :eucl, nlv, K = 5, kavg = 1, h = 1, criw = 4, 
        squared::Bool = false, tolw = 1e-4, scal::Bool = false)
    par = (nlvdis = nlvdis, metric, nlv, K, kavg, h, criw, squared, tolw, scal)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    taby = tab(ycla)
    lev = taby.keys        # class levels (prototypes)
    ni = taby.vals         # number of observations in each prototype class
    nproto = length(lev)   # nb. prototypes
    ## For distances computations
    if par.nlvdis == 0
        fitm_emb = nothing
    else
        fitm_emb = plskern(X, Y; nlv = par.nlvdis)
    end
    ## To store the prototypes (in this version = class centroids)
    Xproto = similar(X, nproto, nco(X))
    Yproto = similar(Y, nproto, nco(Y))
    ## To store the prototype models
    fitm = list(Plsr, nproto)
    coefs = list(NamedTuple, nproto)
    ## Optimize/fit/store the nproto prototype models 
    #@inbounds for i in eachindex(lev)
    Threads.@threads for i in eachindex(lev)
        ind = ycla .== lev[i]
        vX = vrow(X, ind)
        vY = vrow(Y, ind)
        Xproto[i, :] .= colmean(vX)
        Yproto[i, :] .= colmean(vY)
        segm = segmkf(ni[i], K; rep = 1, seed = 1234)
        pars = mpar(scal = scal)
        model = plskern()
        ## To do: adapt for multivariate Y
        rescv = gridcv(model, vX, vY; segm, score = rmsep, pars, nlv = 0:nlv).res
        ## End
        u = findall(rescv.y1 .== minimum(rescv.y1))[1]
        fitm[i] = plskern(vX, vY; nlv = rescv.nlv[u], scal = rescv.scal[u])
        coefs[i] = coef(fitm[i])
    end
    Protoyclaplsr(fitm, fitm_emb, Xproto, Yproto, coefs, ni, lev, par) 
end

function predict(object::Protoyclaplsr, X)
    Q = eltype(object.Xproto)
    X = ensure_mat(X)
    Q = eltype(X)
    m = nro(X)
    q = nco(object.Yproto)
    nproto = nro(object.Xproto)
    ## Params to average the prototype predictions
    kavg = min(object.par.kavg, nproto)  
    h = Q(object.par.h)
    criw = Q(object.par.criw)
    squared = object.par.squared
    tolw = Q(object.par.tolw)
    ## Find the kavg closest prototypes for each new observation
    if isnothing(object.fitm_emb)
        res = getknn(object.Xproto, X; k = kavg, metric = object.par.metric)
    else
        Tproto = transf(object.fitm_emb, object.Xproto)
        T = transf(object.fitm_emb, X)
        if object.par.metric == :mah
            S = covm(object.fitm_emb.T)
            Uinv = LinearAlgebra.inv!(cholesky!(Hermitian(S)).U)
            #Uinv = Diagonal(1 ./ sqrt.(diag(S)))
            res = getknn(Tproto * Uinv, T * Uinv; k = kavg, metric = :eucl)
        else
            res = getknn(Tproto, T; k = kavg, metric = object.par.metric)
        end 
    end
    listnn = res.ind
    listw = list(Vector{Q}, m)
    pred = zeros(m, q)
    ## Compute the kavg protype predictions and average them (weighted average)
    #@inbounds for i = 1:m
    Threads.@threads for i = 1:m   
        s = listnn[i]
        w = winvs(res.d[i]; h, criw, squared)
        @. w[w < tolw] = tolw
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



