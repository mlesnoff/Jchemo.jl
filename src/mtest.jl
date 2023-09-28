""" 
    mtest(Y::DataFrame, id = 1:nro(Y); ntest, 
        rep = 1)
Random splitting of each column of a dataset to a training and test set.
* `Y` : DataFrame (n, p) whose each column (typically, response variables 
    to predict) can contain missing values.
* `id` : Vector (n) of IDs.
* `ntest` : Nb. test observations selected for each `Y` column. 
    The selection is done within the non-missing observations 
    of the considered column. If `ntest` is a single value, the same nb.  
    observations are selected for each column. Alternatively, `ntest` can 
    be a vector of length p. 
* `rep` : Nb. replications of the splitting for each `Y` column.

## Examples
```julia
using DataFrames

Y = hcat([rand(5); missing; rand(6)],
   [rand(2); missing; missing; rand(7); missing])
Y = DataFrame(Y, :auto)
n = nro(Y)

res = mtest(Y; ntest = 3, rep = 4) ;
#res = mtest(Y; ntest = [3; 6], rep = 4) ;
#res = mtest(Y, string.(1:n); ntest = 3, rep = 4) ;
pnames(res)
res.nam
length(res.test)
i = 1    # variable i
res.train[i]
res.test[i]
```
"""
function mtest(Y::DataFrame, id = 1:nro(Y); ntest,
        typ = "rand")
    @assert in(["rand"; "sys"])(typ) "Wrong value for argument 'typ'."
    nam = names(Y)
    p = length(nam)
    idtrain = list(p, Vector)  
    idtest = list(p, Vector)
    length(ntest) == 1 ? ntest = repeat([ntest], p) : nothing
    for i = 1:p
        y = Y[:, nam[i]]
        s_all = findall(ismissing.(y) .== 0)
        ntot = length(s_all)
        ntrain = ntot - ntest[i]
        if typ == "rand"   
            res = samprand(ntot, ntrain)
            idtrain[i] = sort(id[s_all[res.train]])
            idtest[i] = sort(id[s_all[res.test]])
        elseif typ = "sys"
            res = sampsys(y[s_all], ntest[i])
            idtrain[i] = sort(id[s_all[res.test]])
            idtest[i] = sort(id[s_all[res.train]])  
        end    
    end
    (train = idtrain, test = idtest, nam)
end

