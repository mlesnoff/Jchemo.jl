"""
    gridscore(mod, Xtrain, Ytrain, X, Y; 
        score, pars = nothing, nlv = nothing, 
        lb = nothing, verbose = false) 
Model validation over a grid of parameters.
* `mod` : Model to evaluate.
* `Xtrain` : Training X-data (n, p).
* `Ytrain` : Training Y-data (n, q).
* `X` : Validation X-data (m, p).
* `Y` : Validation Y-data (m, q).
Keyword arguments: 
* `score` : Function computing the prediction 
    score (e.g. `rmsep`).
* `pars` : tuple of named vectors of same length defining 
    the parameter combinations (e.g. output of function `mpar`).
* `verbose` : If true, fitting information are printed.
* `nlv` : Value, or vector of values, of the nb. of latent
    variables (LVs).
* `lb` : Value, or vector of values, of the ridge 
    regularization parameter "lambda".

Compute a prediction score (= error rate) for a given model 
over a grid of parameters. The score is computed over the validation 
sets `X` and `Y` for each combination of the grid defined in `pars`. 
    
For models based on LV or ridge regularization, using arguments `nlv` 
and `lb` allow faster computations than including these parameters in 
argument `pars. See the examples.   

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

