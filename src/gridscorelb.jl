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
    q = nco(Ytrain)
    lb = mlev(lb)
    le_lb = length(lb)
    if isnothing(pars)
        verbose ? println("-- Nb. combinations = 0.") : nothing
        fm = fun(Xtrain, Ytrain, lb = maximum(lb))
        pred = Jchemo.predict(fm, X; lb = lb).pred
        le_lb == 1 ? pred = [pred] : nothing
        res = zeros(le_lb, q)
        @inbounds for i = 1:le_lb
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(lb = lb)
    else
        ncomb = length(pars[1])  # nb. combinations in pars
        verbose ? println("-- Nb. combinations = ", ncomb) : nothing
        res = map(values(pars)...) do v...
            verbose ? println(Pair.(keys(pars), v)...) : nothing
            fm = fun(Xtrain, Ytrain ; lb = maximum(lb), Pair.(keys(pars), v)...)
            pred = Jchemo.predict(fm, X; lb = lb).pred
            le_lb == 1 ? pred = [pred] : nothing
            zres = zeros(le_lb, q)
            for i = 1:le_lb
                zres[i, :] = score(pred[i], Y)
            end
            zres
        end 
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

