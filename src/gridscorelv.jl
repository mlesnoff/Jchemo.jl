"""
    gridscorelv(Xtrain, Ytrain, X, Y; score, fun, pars, nlv, verbose = FALSE)
* See `gridscore`.
* `nlv` : Nb., or collection of nb., of latent variables (LVs).

Same as [`gridscore`](@ref) but specific to (and much faster for) models 
using latent variables (e.g. PLSR).

Argument `pars` must not contain `nlv`.

See `?gridscore` for examples.
"""
function gridscorelv(Xtrain, Ytrain, X, Y; score, fun, 
        pars = nothing, nlv, verbose = false)
    Q = eltype(Xtrain[1, 1])
    p = nco(Xtrain)
    q = nco(Ytrain)
    nlv = max(0, minimum(nlv)):min(p, maximum(nlv))
    le_nlv = length(nlv)
    if isnothing(pars)    # e.g.: case of PLSR
        verbose ? println("-- Nb. combinations = 0.") : nothing
        fm = fun(Xtrain, Ytrain, par = Par(nlv = maximum(nlv)))
        pred = predict(fm, X; nlv = nlv).pred
        le_nlv == 1 ? pred = [pred] : nothing
        res = zeros(Q, le_nlv, q)
        @inbounds for i = 1:le_nlv
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(nlv = nlv)
    else
        listpar = [Par(; Dict(kws)...) for 
            kws in zip([[k=>vv for vv in v] for (k, v) in pairs(pars)]...)]
        ncomb = length(listpar)  # nb. combinations in pars
        res = list(ncomb)
        verbose ? println("-- Nb. combinations = ", ncomb) : nothing
        @inbounds for i = 1:ncomb
            verbose ? println(convert(NamedTuple, dat[i, :])) : nothing 
            listpar[i].nlv = maximum(nlv)
            fm = fun(Xtrain, Ytrain; par = listpar[i])
            pred = predict(fm, X; nlv = nlv).pred
            zres = zeros(Q, le_nlv, q)
            @inbounds for j = 1:le_nlv
                zres[j, :] = score(pred[j], Y)
            end
            res[i] = zres
        end
        verbose ? println("-- End.") : nothing
        ncomb == 1 ? res = res[1] : res = reduce(vcat, res) 
        ## Make dat
        if le_nlv == 1
            dat = DataFrame(pars)
        else
            zdat = DataFrame(pars)
            dat = list(ncomb)
            @inbounds for i = 1:ncomb
                dat[i] = reduce(vcat, fill(zdat[i:i, :], le_nlv))
            end
            dat = reduce(vcat, dat)
        end
        znlv = repeat(nlv, ncomb)
        dat = hcat(dat, DataFrame(nlv = znlv))
        ## End
    end
    verbose ? println("-- End.") : nothing
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end

