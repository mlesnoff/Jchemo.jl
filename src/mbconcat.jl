

"""
    mbconcat()
    mbconcat(Xbl)
Concatenate horizontaly multiblock X-data.
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  

## Examples
```julia
n = 5 ; m = 3 ; p = 10 
X = rand(n, p) 
Xnew = rand(m, p)
listbl = [3:4, 1, [6; 8:10]]
Xbl = mblock(X, listbl) 
Xblnew = mblock(Xnew, listbl) 
@head Xbl[3]

mo = mbconcat() 
fit!(mo, Xbl)
transf(mo, Xbl)
transf(mo, Xblnew)
```
"""
function mbconcat(Xbl)
    Mbconcat(nothing)
end

""" 
    transf(object::Mbconcat, Xbl)
Compute the preprocessed data from a model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
""" 
transf(object::Mbconcat, Xbl) = reduce(hcat, Xbl)



