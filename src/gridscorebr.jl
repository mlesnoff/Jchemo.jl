function gridscorebr(Xtrain, Ytrain, X, Y; score, fun, 
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

