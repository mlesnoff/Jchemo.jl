# Not exported

function protoyclaplsr(X, Y, ycla::Vector{String}; nlvdis::Int = 0, metric::Symbol = :eucl, 
        nlv::Int, K::Int = 5, 
        kavg::Int = 1, h::Q = 1., criw::Q = 4., squared::Bool = false, 
        tolw::Q = 1e-4, scalx::Symbol = :none, scaly::Symbol = :none) where {Q<:Float}
    par = (nlvdis = nlvdis, metric, nlv, K, kavg, h, criw, squared, tolw, scalx, scaly)
    X = ensure_mat(X)
    Y = ensure_mat(Y) 
    q = nco(Y)
    taby = tab(ycla)
    lev = taby.lev         # classes (each class = one prototype)
    ni = taby.n            # number of observations in each class
    nproto = length(lev)   # nb. prototypes (= nb. classes in ycla)
    # For distances computations
    if par.nlvdis == 0
        fitm_emb = nothing
    else
        fitm_emb = plskern(X, Y; nlv = par.nlvdis, scalx = par.scalx, scaly = par.scaly)
    end
    # Prepare the fit and the storage of the nproto prototype models 
    fitm = list(Plsr, nproto)
    coefs = list(NamedTuple, nproto)
    # Will store the prototype centers (in this version, = class centroids)
    Xproto = similar(X, nproto, nco(X))
    Yproto = similar(Y, nproto, nco(Y))
    #@inbounds for i in eachindex(lev)
    Threads.@threads for i in eachindex(lev)
        ind = ycla .== lev[i]
        vX = vrow(X, ind)
        vY = vrow(Y, ind)
        Xproto[i, :] .= colmean(vX)
        Yproto[i, :] .= colmean(vY)
        segm = segmkf(ni[i], K; rep = 1, seed = 1234)
        pars = mpar(scalx = [par.scalx], scaly = [par.scaly])
        model = plskern()
        rescv = gridcv(model, vX, vY; segm, score = rmsep, pars, nlv = 0:nlv).res
        # Eventual multivariate Y
        if q == 1
            vscor = rescv.y1
        else
            s = nco(rescv)
            A = Matrix(rescv[:, (s - q + 1):end])
            vscor = sqrt.(rowsum(A.^2))
        end
        u = findall(vscor .== minimum(vscor))[1]
        # Fit and store the optimized model
        fitm[i] = plskern(vX, vY; nlv = rescv.nlv[u], scalx = rescv.scalx[u], scaly = rescv.scaly[u])
        coefs[i] = coef(fitm[i])
    end
    # Outputs
    Protoyclaplsr(fitm, fitm_emb, Xproto, Yproto, coefs, ni, lev, par) 
end

function predict(object::Protoyclaplsr, X)
    X = ensure_mat(X)
    m = nro(X)
    Q = eltype(object.Xproto)
    q = nco(object.Yproto)
    nproto = nro(object.Xproto)
    metric = object.par.metric
    kavg = min(object.par.kavg, nproto)  # nb prototype models to average
    h = object.par.h
    criw = object.par.criw
    squared = object.par.squared
    tolw = object.par.tolw
    # Compute the neighborhood of size kavg (inside dataset Xproto) of each new observation
    if isnothing(object.fitm_emb)
        if object.par.scalx != :none
            csx = cscale(object.Xproto; centr = :none, scal = object.par.scalx)
            X1 = fscale(object.Xproto, csx.scales)
            X2 = fscale(X, csx.scales)
            res = getknn(X1, X2; metric)
        else
            res = getknn(object.Xproto, X; metric, k = kavg)
        end
    else
        # Since nproto is low, a procedure is used to prevent singularity 
        # when 'metric = :mah'  
        if metric == :mah
            Tproto = transf(object.fitm_emb, object.Xproto)
            Tnew = transf(object.fitm_emb, X)
            S = covm(object.fitm_emb.T)
            Uinv = LinearAlgebra.inv!(cholesky!(Hermitian(S)).U)
            #Uinv = Diagonal(1 ./ sqrt.(diag(S)))
            res = getknn(Tproto * Uinv, Tnew * Uinv; k = kavg, metric = :eucl)
        else
            T = transf(object.fitm_emb, object.Xproto)
            res = getknn(T, transf(object.fitm_emb, X); metric, k = kavg)
        end     
    end
    listnn = res.ind
    # Compute the predictions
    listw = list(Vector{Q}, m)
    pred = zeros(Q, m, q)
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



