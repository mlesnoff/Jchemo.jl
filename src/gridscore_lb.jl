"""
    gridscore_lb(Xtrain, Ytrain, X, Y; algo, score, pars = nothing, lb, verbose = false)
Working function for `gridscore`.

Specific, and faster than `gridscore_br`, for models using ridge regularization (e.g., RR). 
Argument `pars` must not contain `lb`.

See function `gridscore` for examples.
"""
function gridscore_lb(Xtrain, Ytrain, X, Y; algo, score, pars = nothing, lb, verbose = false)
    Q = eltype(Xtrain[1, 1])
    q = nco(Ytrain)
    lb = mlev(lb)
    le_lb = length(lb)
    if isnothing(pars)    # e.g.: case of RR
        if verbose ; println("-- Nb. combinations = 0.") ; end
        fitm = algo(Xtrain, Ytrain, lb = maximum(lb))
        pred = predict(fitm, X; lb = lb).pred
        if le_lb == 1 ; pred = [pred] ; end
        res = zeros(Q, le_lb, q)
        @inbounds for i in eachindex(lb) 
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(lb = lb)
    else
        ncomb = length(pars[1])  # nb. combinations in pars
        if verbose ; println("-- Nb. combinations = ", ncomb) ; end
        res = map(values(pars)...) do v...
            if verbose ; println(Pair.(keys(pars), v)...) ; end
            fitm = algo(Xtrain, Ytrain ; lb = maximum(lb), Pair.(keys(pars), v)...)
            pred = predict(fitm, X; lb = lb).pred
            if le_lb == 1 ; pred = [pred] ; end
            zres = zeros(Q, le_lb, q)
            @inbounds for i in eachindex(lb)
                zres[i, :] = score(pred[i], Y)
            end
            zres
        end 
        res = ncomb == 1 ? res[1] : reduce(vcat, res) 
        ## Make dat
        if le_lb == 1
            dat = DataFrame(pars)
        else
            zdat = DataFrame(pars)
            dat = list(ncomb)
            @inbounds for i in eachindex(dat) 
                dat[i] = reduce(vcat, fill(zdat[i:i, :], le_lb))
            end
            dat = reduce(vcat, dat)
        end
        zlb = repeat(lb, ncomb)
        dat = hcat(dat, DataFrame(lb = zlb))
        ## End
    end
    if verbose ; println("-- End.") ; end
    namy = map(string, fill("y", q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end
