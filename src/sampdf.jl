""" 
    sampdf(Y::DataFrame, k::Union{Int, Vector{Int}}, 
        id = 1:nro(Y); msamp = :rand)
Build training vs. test sets from each column of a dataframe. 
* `Y` : DataFrame (n, p) whose each column can contain missing values.
* `k` : Nb. of test observations selected for each `Y` column. 
    The selection is done within the non-missing observations 
    of the considered column. If `k` is a single value, the same nb.  
    of observations are selected for each column. Alternatively, 
    `k` can be a vector of length p. 
* `id` : Vector (n) of IDs.
Keyword arguments:
* `msamp` : Type of sampling for the test set.
    Possible values are: `:rand` = random sampling, 
    `:sys` = systematic sampling over each sorted 
    `Y` column (see function `sampsys`).  

Typically, dataframe `Y` contains a set of response variables 
to predict.

## Examples
```julia
using DataFrames

Y = hcat([rand(5); missing; rand(6)],
   [rand(2); missing; missing; rand(7); missing])
Y = DataFrame(Y, :auto)
n = nro(Y)

k = 3
res = sampdf(Y, k) 
#res = sampdf(Y, k, string.(1:n))
pnames(res)
res.nam
length(res.test)
res.train
res.test

## Replicated splitting Train/Test
rep = 10
k = 3
ids = [sampdf(Y, k) for i = 1:rep]
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
function sampdf(Y::DataFrame, k::Union{Int, Vector{Int}}, 
        id = 1:nro(Y); msamp = :rand)
    @assert in([:rand; :sys])(msamp) "Wrong value for argument 'msamp'."
    p = nco(Y)
    nam = names(Y)
    train = list(Vector, p)  
    test = list(Vector, p)
    length(k) == 1 ? k = repeat([k], p) : nothing
    @inbounds for i = 1:p
        y = Y[:, nam[i]]
        s_all = findall(ismissing.(y) .== 0)
        if msamp == :rand   
            n = length(s_all)
            res = samprand(n, k[i])
        else
            res = sampsys(y[s_all], k[i])
        end 
        ## Sorting
        train[i] = sort(id[s_all[res.train]])
        test[i] = sort(id[s_all[res.test]])             
    end
    (train = train, test, nam)
end

