function gridscorebr(Xtrain, Ytrain, X, Y; score, fun, 
       pars, verbose = false)
    q = nco(Ytrain)
    ncomb = length(pars[1]) # nb. combinations in pars
    verbose ? println("-- Nb. combinations = ", ncomb) : nothing
    res = map(values(pars)...) do v...
        verbose ? println(Pair.(keys(pars), v)...) : nothing
        fm = fun(Xtrain, Ytrain; Pair.(keys(pars), v)...)
        pred = Jchemo.predict(fm, X).pred
        score(pred, Y)
    end
    verbose ? println("-- End.") : nothing
    ncomb == 1 ? res = res[1] : res = reduce(vcat, res) 
    dat = DataFrame(pars)
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end

function gridscorebr_par(Xtrain, Ytrain, X, Y; score, fun, 
        pars, verbose = false)
    q = nco(Ytrain)
    listpar = [Par(; Dict(kws)...) for 
        kws in zip([[k=>vv for vv in v] for (k, v) in pairs(pars)]...)]
    ncomb = length(listpar)  # nb. combinations in pars
    dat = DataFrame(pars)
    res = list(ncomb)
    verbose ? println("-- Nb. combinations = ", ncomb) : nothing
    @inbounds for i = 1:ncomb
        verbose ? println(convert(NamedTuple, dat[i, :])) : nothing 
        fm = fun(Xtrain, Ytrain; par = listpar[i])
        pred = predict(fm, X).pred
        res[i] = score(pred, Y)
    end
    verbose ? println("-- End.") : nothing
    ncomb == 1 ? res = res[1] : res = reduce(vcat, res) 
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end

