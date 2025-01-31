"""
    rasvd(; kwargs...)
    rasvd(X, Y; kwargs...)
    rasvd(X, Y, weights::Weight; kwargs...)
    rasvd!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Redundancy analysis (RA), *aka* PCA on instrumental variables (PCAIV)
* `X` : First block of data.
* `Y` : Second block of data.
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `bscal` : Type of block scaling. Possible values are:
    `:none`, `:frob`. See functions `blockscal`.
* `tau` : Regularization parameter (∊ [0, 1]).
* `scal` : Boolean. If `true`, each column of blocks `X` 
    and `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).
 
See e.g. Bougeard et al. 2011a,b and Legendre & Legendre 2012. 
Let Y_hat be the fitted values of the regression of `Y` on `X`. 
The scores `Ty` are the PCA scores of Y_hat. The scores `Tx` are 
the fitted values of the regression of `Ty` on `X`.

A continuum regularization is available.  After block 
centering and scaling, the covariances matrices are computed 
as follows: 
* Cx = (1 - `tau`) * X'DX + `tau` * Ix
where D is the observation (row) metric. 
Value `tau` = 0 can generate unstability when inverting 
the covariance matrices. Often, a better alternative is 
to use an epsilon value (e.g. `tau` = 1e-8) to get similar 
results as with pseudo-inverses.    

## References
Bougeard, S., Qannari, E.M., Lupo, C., Chauvin, C., 
2011-a. Multiblock redundancy analysis from a user's 
perspective. Application in veterinary epidemiology. 
Electronic Journal of Applied Statistical Analysis 
4, 203-214. https://doi.org/10.1285/i20705948v4n2p203

Bougeard, S., Qannari, E.M., Rose, N., 2011-b. Multiblock 
redundancy analysis: interpretation tools and application 
in epidemiology. Journal of Chemometrics 25, 
467-475. https://doi.org/10.1002/cem.1392

Legendre, V., Legendre, L., 2012. Numerical Ecology. 
Elsevier, Amsterdam, The Netherlands.

Tenenhaus, A., Guillemot, V. 2017. RGCCA: Regularized 
and Sparse Generalized Canonical Correlation Analysis 
for Multiblock Data Multiblock data analysis.
https://cran.r-project.org/web/packages/RGCCA/index.html 

## Examples
```julia
using Jchemo, JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "linnerud.jld2") 
@load db dat
pnames(dat)
X = dat.X
Y = dat.Y
n, p = size(X)
q = nco(Y)

nlv = 2
bscal = :frob ; tau = 1e-4
model = rasvd(; nlv, bscal, tau)
fit!(model, X, Y)
pnames(model)
pnames(model.fitm)

@head model.fitm.Tx
@head transfbl(model, X, Y).Tx

@head model.fitm.Ty
@head transfbl(model, X, Y).Ty

res = summary(model, X, Y) ;
pnames(res)
res.explvarx
res.cort2t 
res.rdx
res.rdy
res.corx2t 
res.cory2t 
```
"""
rasvd(; kwargs...) = JchemoModel(rasvd, nothing, kwargs)

function rasvd(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    n = nro(X)
    weights = mweight(ones(Q, n))
    rasvd(X, Y, weights; kwargs...)
end

function rasvd(X, Y, weights::Weight; kwargs...)
    rasvd!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function rasvd!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParRasvd, kwargs).par 
    @assert in([:none, :frob])(par.bscal) "Wrong value for argument 'bscal'."
    @assert 0 <= par.tau <= 1 "tau must be in [0, 1]"
    Q = eltype(X)
    p = nco(X)
    q = nco(Y)
    nlv = min(par.nlv, p, q)
    tau = convert(Q, par.tau)
    sqrtw = sqrt.(weights.w)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        fcscale!(X, xmeans, xscales)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
    end
    par.bscal == :none ? bscales = ones(Q, 2) : nothing
    if par.bscal == :frob
        normx = frob(X, weights)
        normy = frob(Y, weights)
        X ./= normx
        Y ./= normy
        bscales = [normx ; normy]
    end
    # Row metric
    X .= fweight(X, sqrtw)
    Y .= fweight(Y, sqrtw)
    # End
    if tau == 0       
        invCx = inv(X' * X)
    else
        Ix = Diagonal(ones(Q, p)) 
        if tau == 1   
            invCx = Ix
        else
            invCx = inv((1 - tau) * X' * X + tau * Ix)
        end
    end
    Bx = invCx * X' * Y 
    Yfit = X * Bx
    res = LinearAlgebra.svd(Yfit)
    Wy = res.V[:, 1:nlv]    # = C
    lambda = res.S[1:nlv].^2
    Ty = Y * Wy
    Tx = Yfit * Wy    
    # Same as:
    # Projx = X * invCx * X'
    # Yfit = Projx * Y
    # PCA(Yfit) ==> Wy, Ty
    # Tx = Projx * Ty
    # End
    Tx .= (1 ./ sqrtw) .* Tx
    Ty .= (1 ./ sqrtw) .* Ty   
    Rasvd(Tx, Ty, Bx, Wy, lambda, bscales, xmeans, xscales, ymeans, yscales, 
        weights, par)
end

""" 
    transfbl(object::Rasvd, X, Y; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transfbl(object::Rasvd, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    X = fcscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = fcscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    Yfit = X * object.Bx
    Wy = vcol(object.Wy, 1:nlv)
    Tx = Yfit * Wy
    Ty = Y * Wy
    (Tx = Tx, Ty)
end

## Same as ::Cca
## But explvary has to be computed (To Do)
"""
    summary(object::Rasvd, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::Rasvd, X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = fcscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = fcscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    D = Diagonal(object.weights.w)
    ## X
    T = object.Tx
    sstot = frob(X, object.weights)^2
    tt = diag(T' * D * X * X' * D * T) ./ diag(T' * D * T)
    pvar =  tt / sstot
    cumpvar = cumsum(pvar)
    xvar = tt / n
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## To do: explvary 
    ## Y
    #T .= object.Ty
    #sstot = frob2(Y, object.weights)
    #tt = diag(T' * D * Y * Y' * D * T) ./ diag(T' * D * T)
    #pvar =  tt / sstot
    #cumpvar = cumsum(pvar)
    #explvary = DataFrame(nlv = 1:nlv, var = tt, 
    #    pvar = pvar, cumpvar = cumpvar)
    explvary = nothing 
    ## Correlation between X- and 
    ## Y-block scores
    z = diag(corm(object.Tx, object.Ty, object.weights))
    cort2t = DataFrame(lv = 1:nlv, cor = z)
    ## Redundancies (Average correlations) 
    ## Rd(X, tx) and Rd(Y, ty)
    z = rd(X, object.Tx, object.weights)
    rdx = DataFrame(lv = 1:nlv, rd = vec(z))
    z = rd(Y, object.Ty, object.weights)
    rdy = DataFrame(lv = 1:nlv, rd = vec(z))
    ## Correlation between block variables 
    ## and their block scores
    z = corm(X, object.Tx, object.weights)
    corx2t = DataFrame(z, string.("lv", 1:nlv))
    z = corm(Y, object.Ty, object.weights)
    cory2t = DataFrame(z, string.("lv", 1:nlv))
    ## End 
    (explvarx = explvarx, explvary, cort2t, rdx, rdy, corx2t, cory2t)
end