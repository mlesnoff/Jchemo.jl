"""
    viperm!(model, X, Y; perm::Symbol = :val, score::Function = rmsep, rep::Int = 50, 
        psamp::Float = .3, seed::Union{Nothing, Int} = nothing)
    viperm!(model, X, Y, group::Vector{String}; perm::Symbol = :val, score::Function = rmsep, 
        rep::Int = 50, psamp::Float = .3, seed::Union{Nothing, Int} = nothing)
Permutation variable importance.
* `model` : Model to evaluate.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).  
* `group` : A vector (n) defining groups of observations. Must be a Vector{String}. 
Keyword arguments:
* `perm` : Type of permutation. If `:val` (default), permutations are done on the validation set. 
    If `:cal', they are done one the calibration set. 
* `score` : Function computing the prediction score (an error rate).
* `psamp` : Proportion of observations, or groups (if `group` is used), used as validation set to compute 
    the `score`.
* `rep` : Number of replications of the splitting calibration/validation. 

*Note:* This is an inplace function that modifies the content of object `model` (only).

If `group` is used: for the split calibration/validation (see step-1 below), the function samples groups of observations 
instead of single observations. Such a group-sampling is required when the data are structured by groups and when 
the response to predict is correlated within groups. This prevents underestimation of the generalization error.

The principle is as follows:
1) The observations or groups (rows of data {`X`, `Y`}) are splitted randomly in two sets: a calibration set {Xcal, Ycal}
    and a validation set {Xval, Yval}.
2) Model `model` is fitted on {Xcal, Ycal} and used to compute predictions from Xval. 
    The error rate (`score`) is computed by comparing these predictions to Yval, giving the *reference* 
    error rate.
3) Then, consider vX = Xval if `perm = :val`, or Xcal if `perm = :cal`. The rows of each variable 'j'
    (column of vX) are permuted (rows of other variables than 'j' are not permuted) and the validation error
    is recomputed. The variable importance for 'j' is the difference between this new error rate 
    and the reference error rate.
4) Steps 1-3 are replicated `rep` times. The final variance importances are the averages over
    the `rep` replications.

The outputs returned by the function are:
* `vi` : average results over the `rep` replications.
* `res_rep` : results per replication.

In general, method `perm = :val` returns similar results as the 'out-of-bag' permutation method (Breiman, 2000) 
such as the one used in random forests.

*Warning*: The permutation method is biased for multicolinear variables. Multicolinearity generates 
masking effects. Low values can be returned for variables that are important for the model but that 
are correlated with others (when these variables are permuted, the model has still access to the others 
and the prediction performance is poorly affected).  See for instance an illustration
at 'https://scikit-learn.org/stable/modules/permutation_importance.html#misleading-values-on-strongly-correlated-features',
and in the iris example below.  

# References
Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. https://doi.org/10.1023/A:1010933404324

# Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2")
@load db dat
@names dat
@head dat.X
X = dat.X[:, 2:4]
y = dat.X[:, 1]
ntot = nro(X)
ntest = 30
s = samprand(ntot, ntest)
Xtrain = X[s.train, :]
ytrain = y[s.train]
Xtest = X[s.test, :]
ytest = y[s.test]

# In this example, V2 and V3 are highly collinear, and the most correlated to Y.
# As expected, permutation variance importances are biased (V3 is under-represented, 
# particularly when 'perm = :cal') 

corm(Matrix(Xtrain))
corm(Matrix(Xtrain), ytrain)

model = mlr()
perm = :val
#perm = :cal
res = viperm!(model, Xtrain, ytrain; perm, rep = 200) ;
res.vi
```
"""
function viperm!(model, X, Y; perm::Symbol = :val, score::Function = rmsep, rep::Int = 50, 
        psamp::Float = .3, seed::Union{Nothing, Int} = nothing)
    @assert in(perm, [:val, :cal])  "Wrong value for argument 'perm'. Must be ':val' or ':cal'." 
    X = ensure_mat(X)
    Y = ensure_mat(Y) 
    n, p = size(X)
    q = nco(Y)
    nval = round(Int, psamp * n)
    ncal = n - nval
    Xcal = similar(X, ncal, p)
    Ycal = similar(Y, ncal, q)
    Xval = similar(X, nval, p)
    Yval = similar(Y, nval, q)
    res_rep = similar(X, p, q, rep)
    @inbounds for i = 1:rep
        vseed = isnothing(seed) ? seed : seed + i - 1
        s = samprand(n, nval; seed = vseed)
        Xcal .= X[s.train, :]
        Ycal .= Y[s.train, :]
        Xval .= X[s.test, :]
        Yval .= Y[s.test, :]
        fit!(model, Xcal, Ycal)
        pred = predict(model, Xval).pred
        scor_ref = score(pred, Yval)
        vXcal = similar(Xcal)
        vXval = similar(Xval)
        @inbounds for j = 1:p
            vseed = isnothing(seed) ? seed : seed + j - 1        
            if perm == :val
                vXval .= copy(Xval)
                # Permutation of variable j
                vs = StatsBase.sample(MersenneTwister(vseed), 1:nval, nval; replace = false)
                vXval[:, j] .= vXval[vs, j]
                # End  
                pred .= predict(model, vXval).pred
            else
                vXcal .= copy(Xcal)
                # Permutation of variable j
                vs = StatsBase.sample(MersenneTwister(vseed), 1:ncal, ncal; replace = false)
                vXcal[:, j] .= vXcal[vs, j]
                # End  
                fit!(model, vXcal, Ycal)
                pred .= predict(model, Xval).pred
            end
            res_rep[j, :, i] = score(pred, Yval) - scor_ref
        end
    end
    vi = mean(res_rep, dims = 3)[:, :, 1]
    (vi = vi, res_rep)
end 

function viperm!(model, X, Y, group::Vector{String}; perm::Symbol = :val, score::Function = rmsep, 
        rep::Int = 50, psamp::Float = .3, seed::Union{Nothing, Int} = nothing)
    @assert in(perm, [:val, :cal])  "Wrong value for argument 'perm'. Must be ':val' or ':cal'." 
    X = ensure_mat(X)
    Y = ensure_mat(Y) 
    p = nco(X)
    q = nco(Y)
    lev = mlev(group)
    ngroup = length(lev)
    k = round(Int, psamp * ngroup)
    res_rep = similar(X, p, q, rep)
    @inbounds for i = 1:rep
        vseed = isnothing(seed) ? seed : seed + i - 1
        s = samprand(group, k; seed = vseed)
        Xcal = X[s.train, :]
        Ycal = Y[s.train, :]
        Xval = X[s.test, :]
        Yval = Y[s.test, :]
        nval = nro(Xval)
        fit!(model, Xcal, Ycal)
        pred = predict(model, Xval).pred
        scor_ref = score(pred, Yval)
        vXcal = similar(Xcal)
        vXval = similar(Xval)
        @inbounds for j = 1:p
            vseed = isnothing(seed) ? seed : seed + j - 1        
            if perm == :val
                vXval .= copy(Xval)
                # Permutation of variable j
                vs = StatsBase.sample(MersenneTwister(vseed), 1:nval, nval; replace = false)
                vXval[:, j] .= vXval[vs, j]
                # End  
                pred .= predict(model, vXval).pred
            else
                vXcal .= copy(Xcal)
                # Permutation of variable j
                vs = StatsBase.sample(MersenneTwister(vseed), 1:ncal, ncal; replace = false)
                vXcal[:, j] .= vXcal[vs, j]
                # End  
                fit!(model, vXcal, Ycal)
                pred .= predict(model, Xval).pred
            end
            res_rep[j, :, i] = score(pred, Yval) - scor_ref
        end
    end
    vi = mean(res_rep, dims = 3)[:, :, 1]
    (vi = vi, res_rep)
end 

