"""
    gridscore(Xtrain, Ytrain, X, Y; score, fun, pars, verbose = FALSE) 
Model validation over a grid of parameters.
* `Xtrain` : Training X-data (n, p).
* `Ytrain` : Training Y-data (n, q).
* `X` : Validation X-data (m, p).
* `Y` : Validation Y-data (m, q).
* `score` : Function (e.g. `msep`) computing the prediction score.
* `fun` : Function computing the prediction model.
* `pars` : tuple of named vectors (= arguments of fun) of same length
    involved in the calculation of the score (e.g. output of function `mpar`).
* `verbose` : If true, fitting information are printed.

* `lb` : Value, or collection of values, of the ridge regularization parameter "lambda".

Compute a prediction score (= error rate) for a given model over a grid of parameters.

The score is computed over the validation sets `X` and `Y` for each combination 
of the grid defined in `pars`. 
    
The vectors in `pars` must have same length.

## Examples
```julia
######## Regression 

######## Discrimination

```
"""
function gridscore(mod, Xtrain, Ytrain, X, Y; 
        score, pars = nothing, nlv = nothing, 
        lb = nothing, verbose = false)
    fun = mod.fun
    if isnothing(nlv) && isnothing(lb)
        res = gridscore_br(Xtrain, Ytrain, X, Y; fun, score, 
            pars, verbose)
    elseif !isnothing(nlv)
        res = gridscore_lv(Xtrain, Ytrain, X, Y; fun, score, 
            pars, nlv, verbose)
    elseif !isnothing(lb)
        res = gridscore_lb(Xtrain, Ytrain, X, Y; fun, score, 
            pars, lb, verbose)
    end
    res
end

