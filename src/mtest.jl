""" 
    mtest(Y::DataFrame, id = 1:nro(Y); ntest, 
        typ = "rand")
Split training/test sets for each column of a dataframe 
    (typically, response variables to predict) that can contain missing 
    values
* `Y` : DataFrame (n, p) whose each column can contain missing values.
* `id` : Vector (n) of IDs.
* `ntest` : Nb. test observations selected for each `Y` column. 
    The selection is done within the non-missing observations 
    of the considered column. If `ntest` is a single value, the same nb.  
    observations are selected for each column. Alternatively, `ntest` can 
    be a vector of length p. 
* `typ` : Type of sampling for the test set: "rand" (default) = random sampling, 
    "sys" = systematic sampling (regular grid) over the `Y` column.  

## Examples
```julia
using DataFrames

Y = hcat([rand(5); missing; rand(6)],
   [rand(2); missing; missing; rand(7); missing])
Y = DataFrame(Y, :auto)

mtest(Y; ntest = 3)

mtest(Y; ntest = 3, typ = "sys")

## Replicated splitting Train/Test
rep = 10
ntest = 3
ids = [mtest(Y[:, namy]; ntest = ntest) for i = 1:rep]
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
function mtest(Y::DataFrame, id = 1:nro(Y); ntest, 
        typ = "rand")
    @assert in(["rand"; "sys"])(typ) "Wrong value for argument 'typ'."
    p = nco(Y)
    nam = names(Y)
    idtrain = list(p, Vector)  
    idtest = list(p, Vector)
    length(ntest) == 1 ? ntest = repeat([ntest], p) : nothing
    for i = 1:p
        y = Y[:, nam[i]]
        s_all = findall(ismissing.(y) .== 0)
        if typ == "rand"   
            ntot = length(s_all)
            ntrain = ntot - ntest[i]
            res = samprand(ntot, ntrain)
            idtrain[i] = sort(id[s_all[res.train]])
            idtest[i] = sort(id[s_all[res.test]])
        elseif typ == "sys"
            res = sampsys(y[s_all], ntest[i])
            idtrain[i] = sort(id[s_all[res.test]])
            idtest[i] = sort(id[s_all[res.train]])  
        end    
    end
    (train = idtrain, test = idtest, nam)
end

