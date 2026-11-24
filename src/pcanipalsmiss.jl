"""
    pcanipalsmiss(; kwargs...)
    pcanipalsmiss(X; kwargs...)
    pcanipalsmiss(X::Matrix{Union{Missing, Q}}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
    pcanipalsmiss!(X::Matrix{Union{Missing, Q}}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
PCA by NIPALS algorithm allowing missing data.
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. of principal components (PCs).
* `gs` : Boolean. If `true` (default), a Gram-Schmidt orthogonalization of the scores and loadings is done
    before each X-deflation. 
* `tol` : Tolerance value for stopping the iterations.
* `maxit` : Maximum nb. of iterations.
* `scalx` : Symbol defining the column scaling of `X`. Possible values are: `:none` (default), 
    `:std` (uncorrected STD), `:prt` (Pareto), and `:mad` (MAD).

# References
Wright, K., 2018. Package nipals: Principal Components Analysis using NIPALS with Gram-Schmidt Orthogonalization. 
https://cran.r-project.org/

# Examples
```julia
using Jchemo

X = [1 2. missing 4 ; 4 missing 6 7 ; 
    missing 5 6 13 ; missing 18 7 6 ; 
    12 missing 28 7] 

nlv = 3 
tol = 1e-15
scalx = :none
#scalx = :std
gs = false
#gs = true
model = pcanipalsmiss(; nlv, tol, gs, maxit = 500, scalx)
fit!(model, X)
@names model 
@names fitm = model.fitm
fitm.niter
fitm.csx
fitm.sv
fitm.V
fitm.T

# Orthogonality: only if gs = true
fitm.T' * fitm.T
fitm.V' * fitm.V

# Impute missing data in X
model = pcanipalsmiss(; nlv = 2, gs = true) ;
fit!(model, X)
Xfit = xfit(model.fitm)
s = ismissing.(X)
X_imp = copy(X)
X_imp[s] .= Xfit[s]
X_imp
```
""" 
pcanipalsmiss(; kwargs...) = JchemoModel(pcanipalsmiss, nothing, kwargs)

function pcanipalsmiss(X; kwargs...)
    X = ensure_mat(X)
    v = vec(X)
    s = ismissing.(v) .== 0
    Q = eltype(v[s][1, 1])
    n = nro(X)
    weights = pweight(ones(Q, n))
    pcanipalsmiss(X, weights; kwargs...)
end

function pcanipalsmiss(X::Matrix{Union{Missing, Q}}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
    pcanipalsmiss!(copy(X), weights; kwargs...)
end

function pcanipalsmiss!(X::Matrix{Union{Missing, Q}}, weights::ProbabilityWeights{Q}; kwargs...) where {Q<:Float}
    par = recovkw(ParPcanipals{Q}, kwargs).par
    n, p = size(X)
    nlv = min(n, p, par.nlv)
    par.nlv = nlv
    # Centering and scaling
    centers = colmeanskip(X, weights)
    X .-= centers' 
    scales = ones(Q, p)
    if par.scalx != :none
        scales .= colstdskip(X, weights)
        X ./= scales'
    end
    csx = (centers = centers, scales)
    # End
    sqrtw = sqrt.(weights.values)
    @. X = sqrtw * X
    T = similar(centers, n, nlv)
    V = similar(centers, p, nlv)
    sv = similar(centers, nlv)
    niter = list(Int, nlv)
    if par.gs
        UUt = zeros(Q, n, n)
        VVt = zeros(Q, p, p)
    end
    for a = 1:nlv
        if par.gs == false
            res = nipalsmiss(X; kwargs...)
        else
            res = nipalsmiss(X, UUt, VVt; kwargs...)
        end
        T[:, a] .= res.t
        V[:, a] .= res.v           
        sv[a] = res.sv
        niter[a] = res.niter
        X .-= res.t * res.v'
        if par.gs
            UUt .+= res.u * res.u' 
            VVt .+= res.v * res.v'
        end
    end
    @. T = (1 / sqrtw) * T
    Pcanipals(T, V, sv, csx, weights, niter, par) 
end

