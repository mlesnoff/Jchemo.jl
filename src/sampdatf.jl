""" 
    sampdatf(Y::DataFrame, k::Union{Int, Vector{Int}}, id::Vector = collect(1:nro(Y)); 
        meth::Symbol = :rand, seed::Union{Nothing, Int} = nothing)
Build training vs. test sets from each column of a dataframe. 
* `Y` : DataFrame (n, p). Typivally, contains a set of response variables to predict. Can contain missing values.
* `k` : Nb. of test observations selected for each `Y`-column. The selection is done within the non-missing observations 
    of the considered column. If `k` is a single value, the same nb. of     observations are selected for each column. 
    Alternatively, `k` can be a vector of length p. 
* `id` : Vector (n) of IDs.
Keyword arguments:
* `meth` : Type of sampling for the test set. Possible values are: `:rand` = random sampling, `:sys` = systematic 
    sampling over each sorted `Y`-column (see the principle in function `sampsys`).  
* `seed` : When `meth = :rand`, eventual seed for the `Random.MersenneTwister` generator.

## Examples
```julia
using Jchemo, DataFrames

Y = hcat([rand(5); missing; rand(6)], [rand(2); missing; missing; rand(7); missing])
Y = DataFrame(Y, :auto)
n = nro(Y)

k = 3
res = sampdatf(Y, k) 
#res = sampdatf(Y, k, string.(1:n))
@names res
res.nam
length(res.test)
res.train
res.test

sampdatf(Y, k; seed = 123) 

## Replicated splitting Train/Test
rep = 10
k = 3
ids = [sampdatf(Y, k) for i = 1:rep]
length(ids)
i = 1    # replication
ids[i]
ids[i].train 
ids[i].test
j = 1    # variable y  
ids[i].train[j]
ids[i].test[j]
ids[i].nam[j]
```
"""
function sampdatf(Y::DataFrame, k::Union{Int, Vector{Int}}, id::Vector = collect(1:nro(Y)); 
        meth::Symbol = :rand, seed::Union{Nothing, Int} = nothing)
    @assert in([:rand; :sys])(meth) "Wrong value for argument 'meth'."
    p = nco(Y)
    nam = names(Y)
    Q = eltype(id)
    train = list(Vector{Q}, p)  
    test = list(Vector{Q}, p)
    if length(k) == 1
        k = fill(k, p)
    end
    @inbounds for i in axes(Y, 2)
        y = Y[:, nam[i]]
        s_all = findall(ismissing.(y) .== 0)
        if meth == :rand   
            n = length(s_all)
            res = samprand(n, k[i]; seed)
        else
            res = sampsys(y[s_all], k[i])
        end 
        ## Sorting
        train[i] = sort(id[s_all[res.train]])
        test[i] = sort(id[s_all[res.test]])             
    end
    (train = train, test, nam)
end

