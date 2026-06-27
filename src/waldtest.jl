"""
    waldtest(b, L, varb; h0 = nothing, defden = nothing, digits = 5)
Wald or F test for model coefficients.
* `b` : Vector (p) of the coefficients of the model.
* `L` : Matrix (m, p) such as `L` * `b` gives the linear combination(s) of the coefficients 
    to be tested.
* `varb` : Variance-covariance matrix (p, p) of `b`.
Keyword arguments:
* `h0` : Scalar or vector (m) giving the value(s) of hypothesis H0 to be tested (see below). 
    Default to 0.
* `defden` : Nb. degrees of freedom of the residuals of the model.
* `digits` : Nb. digits for the outputs.

The function tests hypothesis H0: `L` * `b` = `h0`, with either 
* a Chi-squared Wald test (with dfs = m)
* or, if `defden` is given, a F test (with dfs {m, `defden`}).

Both tests assume that `b` is Gaussian.  Compared to the F test, the Wald test neglects the uncertainty 
affecting the estimate of the dispersion parameter of the model (e.g., 'sigma2' in MLRs). 

## References
Diggle, P.J., Liang, K.-Y., Zeger, S.L., 1994. Analysis of longitudinal data. Oxford, Clarendon Press, 253 p.

Draper, N.R., Smith, H., 1998. Applied Regression Analysis. New York, John Wiley & Sons, Inc., 706 p

## Examples 
```julia
using GLM, AnovaGLM

using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/reaction_bertinetto.jld2")
@load db dat
@names dat
datf = dat.datf
n = nro(datf)
tab(datf; group = ["temp", "catal"])  # balanced design

#### Fit of a 2-factor anova model with interaction, using GLM 

contr = EffectsCoding()  # 'sum-to-zero'
#contr = HelmertCoding()
#contr = DummyCoding()   # 'first-level-set-to-0' (default)
contrasts = Dict(:temp => contr, :catal => contr)

f = @formula(y1 ~ 1 + temp + catal + temp & catal)
fitm = lm(f, datf; contrasts)

D = modelmatrix(fitm)       # design matrix
dfm = dof(fitm.model) - 1  # function dof includes the dispersion parameter
dfr = dof_residual(fitm) 
(n = n, dfm, dfr)
b = GLM.coef(fitm)   # model coefficients
varb = vcov(fitm)    # variance-covariance matrix
## Same as:
#s2 = dispersion(fitm.model)^2
#s2 * inv(D' * D)   # = varb

#### Tests
## Hyp. matrices 'L' for contrast 'EffectsCoding' or 'HelmertCoding'
## Not valid for contrasts 'DummyCoding', except the test '4)'

anova(fitm)

## 1) factor 'temp'
L = [0. 1 0 0 0 0 ;
     0 0 1 0 0 0]
waldtest(b, L, varb)                  # Wald test
waldtest(b, L, varb; defden = dfr)  # F test

## 2) factor 'catal'
L = [0. 0 0 1 0 0]
waldtest(b, L, varb) 
waldtest(b, L, varb; defden = dfr)

## 3) interaction 'temp * catal'
L = [0. 0 0 0 1 0 ;
     0 0 0 0 0 1]
waldtest(b, L, varb) 
waldtest(b, L, varb; defden = dfr)

## 4) Whole effect 'catal': 'catal + temp * catal'
L = [0. 0 0 1 0 0 ;
     0 0 0 0 1 0 ;
     0 0 0 0 0 1]
waldtest(b, L, varb) 
waldtest(b, L, varb; defden = dfr)
anova(lm(@formula(y1 ~ 1 + temp), datf; contrasts), fitm)
```
"""
function waldtest(b::Vector{Q}, L::AbstractMatrix{Q}, varb::AbstractMatrix{Q}; 
        h0 = nothing, defden = nothing, digits::Int = 4) where Q <: Float
    dfnum = nro(L)
    h = isnothing(h0) ? L * b : L * b - h0 
    varLb = L * varb * L' 
    val =  h' * inv(varLb) * h
    if isnothing(defden)
        d = Distributions.Chisq(dfnum)
    else
        val = val / dfnum
        d = Distributions.FDist(dfnum, defden)
    end 
    pval = Distributions.ccdf(d, val)
    val = round(val; digits)
    pval = round(pval; digits)
    if isnothing(defden)
        res = (val = val, pval, dfnum)
    else
        res = (val = val, pval, dfnum, defden)
    end
    res
end 
