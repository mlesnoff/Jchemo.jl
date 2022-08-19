struct PlsrAvg
    fm
end

""" 
    plsr_avg(X, Y, weights = ones(size(X, 1)); nlv, 
        typf = "unif", typw = "bisquare", alpha = 0)
    plsr_avg!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); nlv, 
        typf = "unif", typw = "bisquare", alpha = 0, K = 5, rep = 10)
Averaging and stacking PLSR models with different numbers of LVs.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q). Must be univariate (q = 1) if `typw` != "unif".
* `weights` : Weights (n) of the observations.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 
    are averaged). Syntax such as "10" is also allowed ("10": correponds to
    the single model with 10 LVs).
* `typf` : Type of averaging. 

For `typf` in {"aic", "bic", "cv"}
* `typw` : Type of weight function. 
* `alpha` : Parameter of the weight function.

For `typf` = "stack"
* `K` : Nb. of folds segmenting the data in the (K-fold) CV.
* `rep` : Nb. of repetitions of the K-fold CV. 

Ensemblist method where the predictions are computed by averaging 
or stacking the predictions of a set of models built with different numbers of 
latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the average (eventually weighted) or stacking of the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.

Possible values of `typf` are: 
* "unif" : Simple average.
* "aic" : Weighted average based on AIC computed for each model.
* "bic" : Weighted average based on BIC computed for each model.
* "cv" : Weighted average based on RMSEP_CV computed for each model.
* "shenk" : Weighted average using "Shenk et al." weights computed for each model.
* "stack" : Linear stacking. A K-fold CV (eventually repeated) is done and 
the CV predictions are regressed (multiple linear model without intercept)
on the observed response data.

For arguments `typw` and `alpha` (weight function): see `?fweight`.

## References
Lesnoff, M., Roger, J.-M., Rutledge, D.N., 2021. Monte Carlo methods for estimating 
Mallows’s Cp and AIC criteria for PLSR models. Illustration on agronomic spectroscopic NIR data. 
Journal of Chemometrics n/a, e3369. https://doi.org/10.1002/cem.3369

Lesnoff, M., Andueza, D., Barotin, C., Barre, P., Bonnal, L., Fernández Pierna, J.A., Picard, 
F., Vermeulen, P., Roger, J.-M., 2022. Averaging and Stacking Partial Least Squares Regression Models 
to Predict the Chemical Compositions and the Nutritive Values of Forages from Spectral Near 
Infrared Data. Applied Sciences 12, 7850. https://doi.org/10.3390/app12157850

Shenk, J., Westerhaus, M., Berzaghi, P., 1997. Investigation of a LOCAL calibration 
procedure for near infrared instruments. 
Journal of Near Infrared Spectroscopy 5, 223. https://doi.org/10.1255/jnirs.115

Shenk et al. 1998 United States Patent (19). Patent Number: 5,798.526.

Zhang, M.H., Xu, Q.S., Massart, D.L., 2004. Averaged and weighted average partial 
least squares. Analytica Chimica Acta 504, 279–289. https://doi.org/10.1016/j.aca.2003.10.056

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = "5:15"
fm = plsr_avg(Xtrain, ytrain; nlv = nlv) ;
res = predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
f, ax = scatter(vec(res.pred), ytest)
ablines!(ax, 0, 1)
f

fm = plsr_avg(Xtrain, ytrain; nlv = nlv,
    typf = "cv") ;
res = predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
```
""" 
function plsr_avg(X, Y, weights = ones(size(X, 1)); nlv, 
        typf = "unif", typw = "bisquare", alpha = 0, K = 5, rep = 10)
    plsr_avg!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv, 
        typf = typf, typw = typw, alpha = alpha, K = K, rep = rep)
end

function plsr_avg!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); nlv, 
        typf = "unif", typw = "bisquare", alpha = 0, K = 5, rep = 10)
    if typf == "unif"
        fm = plsr_avg_unif!(X, Y, weights; nlv = nlv)
    elseif typf == "aic"
        fm = plsr_avg_aic!(X, Y, weights; nlv = nlv, bic = false,
            typw = typw, alpha = alpha)
    elseif typf == "bic"
        fm = plsr_avg_aic!(X, Y, weights; nlv = nlv, bic = true,
            typw = typw, alpha = alpha)
    elseif typf == "cv"
        fm = plsr_avg_cv!(X, Y, weights; nlv = nlv,
            typw = typw, alpha = alpha)
    elseif typf == "shenk"
        fm = plsr_avg_shenk!(X, Y, weights; nlv = nlv)
    elseif typf == "stack"
        fm = plsr_stack!(X, Y, weights; nlv = nlv, K = K, rep = rep) 
    end
    PlsrAvg(fm)
end

"""
    predict(object::PlsrAvg, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::PlsrAvg, X)
    res = predict(object.fm, X)
    (pred = res.pred, predlv = res.predlv)
end


