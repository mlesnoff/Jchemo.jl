"""
    dfplsr_cg(X, y; nlv, reorth = true, scal = false)
Compute the model complexity (df) of PLSR models with the CGLS algorithm.
* `X` : X-data.
* `y` : Univariate Y-data.
* `nlv` : Nb. latent variables (LVs).
* `reorth` : If `true`, a Gram-Schmidt reorthogonalization of the normal equation 
    residual vectors is done.
* `scal` : Boolean. If `true`, each column of `X` and `y` 
    is scaled by its uncorrected standard deviation.

The number of degrees of freedom (df) of the model is returned for 0, 1, ..., nlv LVs.

## References
Hansen, P.C., 1998. Rank-Deficient and Discrete Ill-Posed Problems, Mathematical Modeling and Computation. 
Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898719697

Hansen, P.C., 2008. Regularization Tools version 4.0 for Matlab 7.3. 
Numer Algor 46, 189–194. https://doi.org/10.1007/s11075-007-9136-9

Lesnoff, M., Roger, J.-M., Rutledge, D.N., 2021. Monte Carlo methods for estimating 
Mallows’s Cp and AIC criteria for PLSR models. Illustration on agronomic spectroscopic NIR data. 
Journal of Chemometrics n/a, e3369. https://doi.org/10.1002/cem.3369

## Examples
```julia
# The example below reproduces the numerical illustration
# given by Kramer & Sugiyama 2011 on the Ozone data (Fig. 1, center).
# Function "pls.model" used for df calculations
# in the R package "plsdof" v0.2-9 (Kramer & Braun 2019)
# automatically scales the X matrix before PLS.
# The example scales X for consistency with plsdof.

using JchemoData, JLD2, DataFrames, CairoMakie 
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "ozone.jld2") 
@load db dat
pnames(dat)

X = dat.X
dropmissing!(X) 
zX = rmcol(Matrix(X), 4) ;
y = X[:, 4] 

# For consistency with plsdof
xstds = colstd(zX)
zXs = scale(zX, xstds)
# End

nlv = 12 
df = dfplsr_cg(zXs, y; nlv = nlv, reorth = true) 
df_kramer = [1.000000, 3.712373, 6.456417, 11.633565, 12.156760, 11.715101, 12.349716,
    12.192682, 13.000000, 13.000000, 13.000000, 13.000000, 13.000000]
f, ax = plotgrid(0:nlv, df_kramer; step = 2,
    xlabel = "Nb. LVs", ylabel = "df")
scatter!(ax, 0:nlv, df.df; color = "red")
ablines!(ax, 1, 1; color = :grey, linestyle = :dot)
f
```
""" 
function dfplsr_cg(X, y; nlv, reorth = true, scal = false)
    F = cglsr(X, y; nlv = nlv, reorth = reorth, filt = true, scal = scal).F
    df = [1 ; vec(1 .+ sum(F, dims = 1))]
    (df = df,)
end

"""
    aicplsr(X, y; nlv, correct = true, bic = false, scal = false)
Compute Akaike's (AIC) and Mallows's (Cp) criteria for univariate PLSR models.
* `X` : X-data.
* `y` : Univariate Y-data.
* `nlv` : Nb. latent variables (LVs).
* `correct` : Define if the bias correction is applied.
* `bic` : Define is BIC is computed instead of AIC.
* `scal` : Boolean. If `true`, each column of `X` and `y` 
    is scaled by its uncorrected standard deviation.

`X` and `y` are internally centered. 

The function uses function `dfplsr_cg`. 

## References
Hansen, P.C., 1998. Rank-Deficient and Discrete Ill-Posed Problems, Mathematical Modeling and Computation. 
Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898719697

Hansen, P.C., 2008. Regularization Tools version 4.0 for Matlab 7.3. 
Numer Algor 46, 189–194. https://doi.org/10.1007/s11075-007-9136-9

Lesnoff, M., Roger, J.-M., Rutledge, D.N., 2021. Monte Carlo methods for estimating 
Mallows’s Cp and AIC criteria for PLSR models. Illustration on agronomic spectroscopic NIR data. 
Journal of Chemometrics n/a, e3369. https://doi.org/10.1002/cem.3369

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc 

nlv = 25
res = aicplsr(X, y; nlv = nlv) 
pnames(res)
res.crit
res.opt
res.delta

zaic = aicplsr(X, y; nlv = nlv).crit.aic
f, ax = plotgrid(0:nlv, zaic;
    xlabel = "Nb. LVs", ylabel = "AIC")
scatter!(ax, 0:nlv, zaic)
f
```
""" 
function aicplsr(X, y; nlv, correct = true, bic = false, scal = false)
    X = ensure_mat(X)
    n = size(X, 1)
    p = size(X, 2)
    nlv = min(nlv, n, p)
    pars = mpar(scal = scal)  
    res = gridscorelv(X, y, X, y;
        fun = plskern, score = ssr, nlv = 0:nlv, pars = pars)
    zssr = res.y1
    df = dfplsr_cg(X, y, nlv = nlv, reorth = true, scal = scal).df
    df_ssr = n .- df
    # For Cp, unbiased estimate of sigma2 
    # ----- Cp1: From a low biased model
    # Not stable with dfcov and nlv too large compared to best model !!
    # If df stays below .95 * n, this corresponds
    # to the maximal model (nlv)
    # Option 2 gives in general results
    # very close to those of option 1,
    # but can give poor results with dfcov
    # when nlv is set too large to the best model
    k = maximum(findall(df .<= .5 * n))
    s2_1 = zssr[k] / df_ssr[k]
    # ----- Cp2: FPE-like
    # s2 is estimated from the model under evaluation
    # Used in Kraemer & Sugiyama 2011 Eq.5-6
    s2_2 = zssr ./ df_ssr
    ct = ones(nlv + 1)
    correct ? ct .= n ./ (n .- df .- 2) : nothing
    ct[(df .> n) .| (ct .<= 0)] .= NaN 
    # For safe predictions when df stabilizes and fluctuates
    ct[df .> .8 * n] .= NaN
    # End
    u = findall(isnan.(ct)) 
    if length(u) > 0
        ct[minimum(u):(nlv + 1)] .= NaN
    end
    bic ? alpha = log(n) : alpha = 2
    aic = n * log.(zssr) + alpha * (df .+ 1) .* ct
    cp1 = zssr .+ 2 * s2_1 * df .* ct
    cp2 = zssr .+ 2 * s2_2 .* df .* ct
    aic ./= n
    cp1 ./= n
    cp2 ./= n
    res = (aic = aic, cp1 = cp1, cp2 = cp2)
    znlv = 0:nlv
    tab = DataFrame(nlv = znlv, n = fill(n, nlv + 1), df = df, ct = ct, ssr = zssr)
    crit = hcat(tab, DataFrame(res))
    opt = map(x -> findmin(x[isnan.(x) .== 0])[2] - 1, res)
    delta = map(x -> x .- findmin(x[isnan.(x) .== 0])[1], res)                  # Differences "Delta"
    delta = reduce(hcat, delta)
    nam = [:aic, :cp1, :cp2]
    delta = DataFrame(delta, nam)
    insertcols!(delta, 1, :nlv => znlv)
    #w = map(x -> exp.(-x / 2) / sum(exp.(-x[isnan.(x) .== 0] / 2)), delta)      # AIC model weights   
    #w = reduce(hcat, w)
    #w = DataFrame(w, nam)
    #insertcols!(w, 1, :nlv => znlv)
    (crit = crit, opt = opt, delta = delta)
end

