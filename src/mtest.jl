""" 
    mtest(Y::DataFrame, id = 1:nro(Y); test, rep = 1)
Select indexes defining training and test sets for each column 
    of a dataframe.
* `Y` : DataFrame (n, p). Typically responses variables to predict. 
    Missing values are allowed.
* `id` : Vector (n) of IDs.
* `test` : Nb. (if Int64) or proportion (if Float64)
    of observations in each test set, within the non-missing 
    observations of the considered `Y` column.
* `rep` : Nb. repetitions of training and test sets for each `Y` column.

## Examples
```julia
using DataFrames

Y = hcat([rand(5); missing; rand(6)],
   [rand(2); missing; missing; rand(7); missing])
Y = DataFrame(Y, :auto)
n = nro(Y)

res = mtest(Y; test = 3, rep = 4) ;
#res = mtest(Y; test = .3, rep = 4) ;
#res = mtest(Y, string.(1:n); test = 3, rep = 4) ;
pnames(res)
res.nam
length(res.test)
length(res.test[1])
i = 1
res.test[i]
res.train[i]
```
"""
function mtest(Y::DataFrame, id = 1:nro(Y); test, rep = 1)
    nam = names(Y)
    nvar = length(nam)
    idtest = list(nvar, Vector)
    idtrain = list(nvar, Vector)  
    for i = 1:nvar
        znam = nam[i]
        y = Y[:, znam]
        s_all = findall(ismissing.(y) .== 0)      
        ntot = length(s_all)
        isa(test, Int64) ? ntest = copy(test) : nothing
        isa(test, Float64) ? ntest = Int64(round(test * ntot)) : nothing
        zidtest = list(rep, Vector)
        zidtrain = list(rep, Vector)
        for j = 1:rep 
            s_test = sample(1:ntot, ntest; replace = false)         
            s_train = (1:ntot)[in(s_test).(1:ntot) .== 0]
            zidtest[j] = sort(id[s_all[s_test]])      
            zidtrain[j] = sort(id[s_all[s_train]])
        end
        idtest[i] = zidtest
        idtrain[i] = zidtrain
    end
    (test = idtest, train = idtrain, nam)
end
