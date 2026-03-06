struct ProtoYclaPlsr
    Xproto::Matrix
    Yproto::Matrix
    fitm::Vector{Plsr}    
    coefs::Vector
    ni::Vector{Int}
    lev::Vector
    par::NamedTuple
end

## Not exported
function protoyclaplsr(X, Y, ycla; metric = :eucl, nlv, kavg = 1, h = 1, criw = 4, squared = false, 
        tolw = 1e-4, scal = false)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    taby = tab(ycla)
    lev = taby.keys        # class levels (prototypes)
    ni = taby.vals         # number of observations in each prototype
    nproto = length(lev)   # nb. prototypes
    ## To store the prototype models
    fitm = list(Plsr, nproto)
    coefs = list(NamedTuple, nproto)
    ## To store the prototype centroids
    Xproto = similar(X, nproto, nco(X))
    Yproto = similar(Y, nproto, nco(Y))
    ## End
    #@inbounds for i in eachindex(lev)
    Threads.@threads for i in eachindex(lev)
        ind = ycla .== lev[i]
        vX = vrow(X, ind)
        vY = vrow(Y, ind)
        ## Store prototype center (in this version, centroid)
        Xproto[i, :] .= colmean(vX)
        Yproto[i, :] .= colmean(vY)
        ## Store prototype model    
        segm = segmkf(ni[i], 4; rep = 1, seed = 1234)
        n = Int(round(ni[i] * 0.66))
        nlv = min(n, nlv)
        pars = mpar(scal = scal)
        model = plskern()
        ## To do: adapt for multivariate Y
        rescv = gridcv(model, vX, vY; segm, score = rmsep, pars, nlv = 0:nlv).res
        ## End
        u = findall(rescv.y1 .== minimum(rescv.y1))[1]
        fitm[i] = plskern(vX, vY; nlv = rescv.nlv[u], scal = rescv.scal[u])
        coefs[i] = coef(fitm[i])
    end
    par = (metric = metric, nlv, kavg, h, criw, squared, tolw, scal)
    ProtoYclaPlsr(Xproto, Yproto, fitm, coefs, ni, lev, par) 
end

function predict(object::ProtoYclaPlsr, X)
    X = ensure_mat(X)
    Q = eltype(X)
    m = nro(X)
    q = nco(object.Yproto)
    nproto = nro(object.Xproto)
    metric = object.par.metric
    ## Params for averaging the prototype predictions
    kavg = min(object.par.kavg, nproto)  # nb prototype models averaged
    h = object.par.h
    criw = object.par.criw
    squared = object.par.squared
    tolw = object.par.tolw
    ## Find the kavg closest prototype centers for each new observation
    res = getknn(object.Xproto, X; k = kavg, metric)
    listnn = res.ind
    listw = list(Vector{Q}, m)
    pred = zeros(m, q)
    ## Compute the kavg protype predictions and average them (weighted average)
    #@inbounds for i = 1:m
    Threads.@threads for i = 1:m   
        s = listnn[i]
        w = winvs(res.d[i]; h, criw, squared)
        w[w .< tolw] .= tolw
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



