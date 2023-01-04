""" 
    mtest(Y::DataFrame, id = 1:nro(Y); ntest, 
        nb = true, rep = 1)
Select indexes defining training and test sets for each column 
    of a dataframe.
* `Y` : DataFrame (n, p). Typically responses variables. 
    Missing values are allowed.
* `id` : Vector (n) of IDs.
* `ntest` : Nb. observations (or proportion, see `nb`) in each test set.
* `nb` : Boolean. If `true`, ntest is a number, else it 
    is the proportion of the nb. observation (non missing) in 
    each column of `Y`.
* `rep` : Nb. repetitions of training and test sets for each column.

The function returns an Int64 array of indexes of size (ntest, rep, p).

## Examples
```julia
using DataFrames

Y = hcat([rand(5); missing; rand(6)],
   [rand(2); missing; missing; rand(7); missing])
Y = DataFrame(Y, :auto)
n = nro(Y)

res = mtest(Y; ntest = 3, rep = 4) ;
#res = mtest(Y, string.(1:n); ntest = 3, rep = 4) ;
pnames(res)
res.nam
length(res.idtest)
length(res.idtest[1])
i = 1
res.idtest[i]
res.idtrain[i]
```
"""
function mtest(Y::DataFrame, id = 1:nro(Y); ntest, 
        nb = true, rep = 1)
    n = nro(Y)
    nam = names(Y)
    nvar = length(nam)
    idtest = list(nvar, Vector)
    idtrain = list(nvar, Vector)
    for i = 1:nvar
        znam = nam[i]
        y = Y[:, znam]
        s_all = findall(ismissing.(y) .== 0)      
        ntot = length(s_all)
        if !nb
            pct = copy(ntest)
            ntest = Int64(round(pct * ntrain))
        end
        ntrain = ntot - ntest
        zidtest = list(rep, Vector)
        zidtrain = list(rep, Vector)
        for j = 1:rep 
            s_test = sample(1:ntot, ntest; replace = false)         
            s_train = (1:ntot)[in(s_test).(1:ntot) .== 0]

            #zidtest[j] = sort(s_all[s_test])      
            #zidtrain[j] = sort(s_all[s_train])
            
            zidtest[j] = sort(id[s_all[s_test]])      
            zidtrain[j] = sort(id[s_all[s_train]])
            
        end
        idtest[i] = zidtest
        idtrain[i] = zidtrain
    end
    (idtest = idtest, idtrain, nam)
end
