"""
    gridscorelb(Xtrain, Ytrain, X, Y; score, fun, lb, pars, verbose = FALSE)
* See `gridscore`.
* `lb` : Value, or collection of values, of the ridge regularization parameter "lambda".

Same as [`gridscore`](@ref) but specific to (and much faster for) models 
using ridge regularization (e.g. RR).

Argument `pars` must not contain `lb`.

See `?gridscore` for examples.
"""
function gridscorelb(Xtrain, Ytrain, X, Y; score, fun, lb, 
        pars = nothing, verbose = false)
    Q = eltype(Xtrain[1, 1])
    q = nco(Ytrain)
    lb = mlev(lb)
    le_lb = length(lb)
    if isnothing(pars)    # e.g.: case of RR
        verbose ? println("-- Nb. combinations = 0.") : nothing
        fm = fun(Xtrain, Ytrain, par = Par(lb = maximum(lb)))
        pred = predict(fm, X; lb = lb).pred
        le_lb == 1 ? pred = [pred] : nothing
        res = zeros(Q, le_lb, q)
        @inbounds for i = 1:le_lb
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(lb = lb)
    else
        listpar = [Par(; Dict(kws)...) for 
            kws in zip([[k=>vv for vv in v] for (k, v) in pairs(pars)]...)]
        ncomb = length(listpar)  # nb. combinations in pars
        res = list(ncomb)
        verbose ? println("-- Nb. combinations = ", ncomb) : nothing
        for i = 1:ncomb
            verbose ? println(convert(NamedTuple, dat[i, :])) : nothing 
            listpar[i].lb = maximum(lb)
            fm = fun(Xtrain, Ytrain; par = listpar[i])
            pred = predict(fm, X; lb = lb).pred
            zres = zeros(Q, le_lb, q)
            @inbounds for j = 1:le_lb
                zres[j, :] = score(pred[j], Y)
            end
            res[i] = zres
        end
        verbose ? println("-- End.") : nothing
        ncomb == 1 ? res = res[1] : res = reduce(vcat, res) 
        ## Make dat
        if le_lb == 1
            dat = DataFrame(pars)
        else
            zdat = DataFrame(pars)
            dat = list(ncomb)
            @inbounds for i = 1:ncomb
                dat[i] = reduce(vcat, fill(zdat[i:i, :], le_lb))
            end
            dat = reduce(vcat, dat)
        end
        zlb = repeat(lb, ncomb)
        dat = hcat(dat, DataFrame(lb = zlb))
        ## End
    end
    verbose ? println("-- End.") : nothing
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end

