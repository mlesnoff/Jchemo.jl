"""
    gridscorelv(Xtrain, Ytrain, X, Y; score, fun, nlv, pars, verbose = FALSE)
* See `gridscore`.
* `nlv` : Nb., or collection of nb., of latent variables (LVs).

Same as [`gridscore`](@ref) but specific to (and much faster for) models 
using latent variables (e.g. PLSR).

Argument `pars` must not contain `nlv`.

See `?gridscore` for examples.
"""
function gridscorelv(Xtrain, Ytrain, X, Y; score, fun, nlv, 
        pars = nothing, verbose = false)
    # If not multiblock
    if isa(Xtrain, Matrix)
        p = nco(Xtrain)
        nlv = max(0, minimum(nlv)):min(p, maximum(nlv))
    end
    # End
    q = nco(Ytrain)
    le_nlv = length(nlv)
    if isnothing(pars)
        verbose ? println("-- Nb. combinations = 0.") : nothing
        fm = fun(Xtrain, Ytrain, nlv = maximum(nlv))
        pred = Jchemo.predict(fm, X; nlv = nlv).pred
        le_nlv == 1 ? pred = [pred] : nothing
        res = zeros(le_nlv, q)
        @inbounds for i = 1:le_nlv
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(nlv = nlv)
    else       
        ncomb = length(pars[1])  # nb. combinations in pars
        verbose ? println("-- Nb. combinations = ", ncomb) : nothing
        res = map(values(pars)...) do v...    
            verbose ? println(Pair.(keys(pars), v)...) : nothing
            fm = fun(Xtrain, Ytrain ; nlv = maximum(nlv), Pair.(keys(pars), v)...)
            pred = Jchemo.predict(fm, X; nlv = nlv).pred
            le_nlv == 1 ? pred = [pred] : nothing
            zres = zeros(le_nlv, q)
            @inbounds for i = 1:le_nlv
                zres[i, :] = score(pred[i], Y)
            end
            zres
        end 
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

