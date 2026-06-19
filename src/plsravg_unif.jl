function plsravg_unif(X, Y; kwargs...)
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    plsravg_unif(X, Y, weights; kwargs...)
end

function plsravg_unif(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    plsravg_unif!(copy(X), copy(Y), weights; kwargs...)
end

function plsravg_unif!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    par = recovkw(ParPlsravgunif{Q}, kwargs).par
    X = ensure_mat(X)
    n, p = size(X)
    nlv = min(n, p, minimum(par.nlv)):min(n, p, maximum(par.nlv))
    par.nlv = nlv
    fitm = plskern!(X, Y, weights; nlv = maximum(nlv), scal = par.scal)
    Plsravgunif(fitm, par) 
end

function predict(object::Plsravgunif, X)
    nlv = object.par.nlv
    le_nlv = length(nlv)
    predlv = predict(object.fitm, X, nlv).pred
    if(le_nlv == 1)
        pred = predlv
    else
        acc = copy(predlv[1])
        @inbounds for i = 2:le_nlv
            acc .+= predlv[i]
        end
        pred = acc / le_nlv
    end
    (pred = pred, predlv)
end


