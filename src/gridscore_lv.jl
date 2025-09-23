"""
    gridscore_lv(Xtrain, Ytrain, X, Y; algo, score, pars = nothing, nlv, verbose = false)
Working function for `gridscore`.

Specific and faster than `gridscore_br` for models using latent variables (e.g. PLSR). Argument `pars` must 
not contain `nlv`.

See function `gridscore` for examples.
"""
function gridscore_lv(Xtrain, Ytrain, X, Y; algo, score, pars = nothing, nlv, verbose = false)
    Q = eltype(Xtrain[1, 1])
    q = nco(Ytrain)
    le_nlv = length(nlv)
    if isnothing(pars)   # e.g.: case of PLSR
        verbose ? println("-- Nb. combinations = 0.") : nothing
        fitm = algo(Xtrain, Ytrain; nlv = maximum(nlv))
        pred = predict(fitm, X; nlv).pred
        le_nlv == 1 ? pred = [pred] : nothing
        res = zeros(Q, le_nlv, q)
        @inbounds for i in eachindex(nlv)
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(nlv = nlv)
    else       
        ncomb = length(pars[1])  # nb. combinations in pars
        verbose ? println("-- Nb. combinations = ", ncomb) : nothing
        res = map(values(pars)...) do v...    
            verbose ? println(Pair.(keys(pars), v)...) : nothing
            fitm = algo(Xtrain, Ytrain ; nlv = maximum(nlv), Pair.(keys(pars), v)...)
            pred = predict(fitm, X; nlv).pred
            le_nlv == 1 ? pred = [pred] : nothing
            zres = zeros(Q, le_nlv, q)
            @inbounds for i in eachindex(nlv)
                zres[i, :] = score(pred[i], Y)
            end
            zres
        end 
        ncomb == 1 ? res = res[1] : 
            res = reduce(vcat, res) 
        ## Make dat
        if le_nlv == 1
            dat = DataFrame(pars)
        else
            zdat = DataFrame(pars)
            dat = list(ncomb)
            @inbounds for i in eachindex(dat) 
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
