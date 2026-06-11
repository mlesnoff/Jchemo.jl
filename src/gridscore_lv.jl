"""
    gridscore_lv(Xtrain, Ytrain, X, Y; algo, score::Function, 
        pars::Union{Nothing, NamedTuple} = nothing, 
        nlv::Union{Int, AbstractVector{Int}}, verbose::Bool = false)
Working function for `gridscore`.

Specific, and faster than `gridscore_br`, for models using latent variables (e.g., PLSR). 
Argument `pars` must not contain `nlv`.

See function `gridscore` for examples.
"""
function gridscore_lv(Xtrain, Ytrain, X, Y; algo, score::Function, 
        pars::Union{Nothing, NamedTuple} = nothing, 
        nlv::Union{Int, AbstractVector{Int}}, verbose::Bool = false)
    ## The function works for mono- and multiblock X
    Q = eltype(Xtrain[1, 1])
    ## Monoblock
    if isa(Xtrain[1, 1], Number)
        n, p = size(Xtrain)
    ## Multiblock
    else  
        n = nco(Xtrain[1])
        p = sum(nco.(Xtrain))
    end
    ## End
    q = nco(Ytrain)
    le_nlv = length(nlv)
    ## Rebuild 'nlv' to ensure consistency with training dimensionality
    if le_nlv == 1
        nlv = minimum([maximum(nlv); n - 1; p - 1])
    else
        nlv = minimum(nlv):minimum([maximum(nlv); n - 1; p - 1])
        le_nlv = length(nlv)
    end
    ## End
    if isnothing(pars)   # e.g.: case of PLSR with no scaling
        if verbose ; println("-- Nb. combinations = 0.") ; end
        fitm = algo(Xtrain, Ytrain; nlv = maximum(nlv))
        pred = predict(fitm, X, nlv).pred
        res = zeros(Q, le_nlv, q)
        @inbounds for i in eachindex(nlv)
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(nlv = nlv)
    else
        ncomb = length(pars[1])  # nb. combinations in pars
        if verbose ; println("-- Nb. combinations = ", ncomb) ; end
        res = map(values(pars)...) do v...    
            if verbose ; println(Pair.(keys(pars), v)...) ; end
            fitm = algo(Xtrain, Ytrain ; nlv = maximum(nlv), Pair.(keys(pars), v)...)
            pred = predict(fitm, X, nlv).pred
            zres = zeros(Q, le_nlv, q)
            @inbounds for i in eachindex(nlv)
                zres[i, :] = score(pred[i], Y)
            end
            zres
        end 
        res = ncomb == 1 ? res[1] : reduce(vcat, res) 
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
    if verbose ; println("-- End.") ; end
    namy = map(string, fill("y", q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end

