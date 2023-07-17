struct Nscda3
    fms
    cts::Array{Float64}
    wprior::Vector{Float64}
    ni::Vector{Int64}
    lev::AbstractVector
    weights::Vector{Float64}
end

function nscda(X, y, weights = ones(nro(X)); delta, 
        prior = "unif", scal::Bool = false)
    weights = mweight(weights)
    fms = nsc(X, y, weights;
        delta = delta, scal = scal)
    poolstd_s0 = fms.poolstd .+ fms.s0
    ni = fms.ni
    lev = fms.lev
    nlev = length(lev)
    if isequal(prior, "unif")
        wprior = ones(nlev) / nlev
    elseif isequal(prior, "prop")
        wprior = mweight(ni)
    end
    Nscda3(fms, poolstd_s0, wprior, ni, 
        lev, weights)
end

function predict(object::Nscda3, X; meth = 2)
    zX = scale(X, object.xscales)
    m = nro(zX)
    scale!(zX, poolstd_s0)
    cts = scale(object.fms.cts, poolstd_s0)
    d2 = euclsq(zX, cts) .- 2 * log.(object.wprior)'
    posterior = softmax(-.5 * d2)
    z =  mapslices(argmin, d2; dims = 2)  # if equal, argmax takes the first
    pred = reshape(replacebylev2(z, object.lev), m, 1)
    (pred = pred, d2, posterior)
end

