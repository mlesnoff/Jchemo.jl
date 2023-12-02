"""
    dfplsr_cg(X, y; nlv, gs = true, scal::Bool = false)
Compute the model complexity (df) of PLSR models with the CGLS algorithm.
* `X` : X-data.
* `y` : Univariate Y-data.
* `nlv` : Nb. latent variables (LVs).
* `gs` : If `true`, a Gram-Schmidt reorthogonalization of the normal equation 
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
# given by Kramer & Sugiyama 2011 on the Ozone data (Fig. 1, fcenter).
# Function "pls.model" used for df calculations
# in the R package "plsdof" v0.2-9 (Kramer & Braun 2019)
# automatically scales the X matrix before PLS.
# The example scales X for consistency with plsdof.

using JchemoData, JLD2, DataFrames, CairoMakie 
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/ozone.jld2") 
@load db dat
pnames(dat)

X = dat.X
dropmissing!(X) 
zX = rmcol(Matrix(X), 4) ;
y = X[:, 4] 

# For consistency with plsdof
xstds = colstd(zX)
zXs = fscale(zX, xstds)
# End

nlv = 12 
df = dfplsr_cg(zXs, y; nlv = nlv, gs = true) 
df_kramer = [1.000000, 3.712373, 6.456417, 11.633565, 12.156760, 11.715101, 12.349716,
    12.192682, 13.000000, 13.000000, 13.000000, 13.000000, 13.000000]
f, ax = plotgrid(0:nlv, df_kramer; step = 2,
    xlabel = "Nb. LVs", ylabel = "df")
scatter!(ax, 0:nlv, df.df; color = "red")
ablines!(ax, 1, 1; color = :grey, linestyle = :dot)
f
```
""" 
function dfplsr_cg(X, y; kwargs...)
    F = cglsr(X, y; kwargs...).F
    df = [1 ; vec(1 .+ sum(F, dims = 1))]
    (df = df,)
end

