""" 
    sampdf(Y::DataFrame, id = 1:nro(Y); k, 
        sampm = :rand)
Build training/test sets for each column of a dataframe 
    (typically, response variables to predict) that can contain missing 
    values
* `Y` : DataFrame (n, p) whose each column can contain missing values.
* `id` : Vector (n) of IDs.
* `k` : Nb. test observations selected for each `Y` column. 
    The selection is done within the non-missing observations 
    of the considered column. If `k` is a single value, the same nb.  
    observations are selected for each column. Alternatively, `k` can 
    be a vector of length p. 
* `sampm` : Type of sampling for the test set: :rand (default) = random sampling, 
    :sys = systematic sampling (regular grid) over the `Y` column.  

## Examples
```julia
using DataFrames

Y = hcat([rand(5); missing; rand(6)],
   [rand(2); missing; missing; rand(7); missing])
Y = DataFrame(Y, :auto)

sampdf(Y; k = 3)

sampdf(Y; k = 3, sampm = :sys)

## Replicated splitting Train/Test
rep = 10
k = 3
ids = [sampdf(Y[:, namy]; k = k) for i = 1:rep]
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
function sampdf(Y::DataFrame, k, id = 1:nro(Y); 
        sampm = :rand)
    @assert in([:rand; :sys])(sampm) "Wrong value for argument 'sampm'."
    p = nco(Y)
    nam = names(Y)
    idtrain = list(p, Vector)  
    idtest = list(p, Vector)
    length(k) == 1 ? k = repeat([k], p) : nothing
    @inbounds for i = 1:p
        y = Y[:, nam[i]]
        s_all = findall(ismissing.(y) .== 0)
        if sampm == :rand   
            n = length(s_all)
            ntrain = n - k[i]
            res = samprand(n, ntrain)
            idtrain[i] = sort(id[s_all[res.train]])
            idtest[i] = sort(id[s_all[res.test]])
        elseif sampm == :sys
            res = sampsys(y[s_all], k[i])
            idtrain[i] = sort(id[s_all[res.test]])
            idtest[i] = sort(id[s_all[res.train]])  
        end    
    end
    (train = idtrain, test = idtest, nam)
end

