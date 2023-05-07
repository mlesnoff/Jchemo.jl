struct Kernda
    fm
    wprior::AbstractVector
    lev::AbstractVector
    ni::AbstractVector
end

function kernda(X, y; prior = "unif", h = nothing, a = 1)
    X = ensure_mat(X)
    lev = mlev(y)
    nlev = length(lev)
    ni = tab(y).vals
    if isequal(prior, "unif")
        wprior = ones(nlev) / nlev
    elseif isequal(prior, "prop")
        wprior = mweight(ni)
    end
    fm = list(nlev)
    for i = 1:nlev
        s = y .== lev[i]
        zX = vrow(X, s)
        fm[i] = dmkern(zX; h = h, a = a)
    end
    Kernda(fm, wprior, lev, ni)
end

function predict(object::Kernda, X)
    X = ensure_mat(X)
    m = nro(X)
    lev = object.lev
    nlev = length(lev) 
    dens = similar(X, m, nlev)
    ni = object.ni
    for i = 1:nlev
        dens[:, i] .= vec(Jchemo.predict(object.fm[i], X).pred)
    end
    A = object.wprior' .* dens
    v = sum(A, dims = 2)
    posterior = scale(A', v)'                    # This could be replaced by code similar as in scale! 
    z =  mapslices(argmax, posterior; dims = 2)  # if equal, argmax takes the first
    pred = reshape(replacebylev2(z, object.lev), m, 1)
    (pred = pred, dens = dens, posterior = posterior)
end
    
