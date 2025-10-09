"""
    cca(; kwargs...)
    cca(X, Y; kwargs...)
    cca(X, Y, weights::Weight; kwargs...)
    cca!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Canonical correlation Analysis (CCA, RCCA).
* `X` : First block of data.
* `Y` : Second block of data.
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs; = scores) to compute.
* `bscal` : Type of block scaling. Possible values are:`:none`, `:frob`. See functions `blockscal`.
* `tau` : Regularization parameter (∊ [0, 1]).
* `scal` : Boolean. If `true`, each column of blocks `X` and `Y` is scaled by its uncorrected standard 
    deviation (before the block scaling).

This function implements a CCA algorithm using SVD decompositions and presented in Weenink 2003 section 2. 

A continuum regularization is available (parameter `tau`). After block centering and scaling, 
the function returns block LVs (Tx and Ty) that are proportionnal to the eigenvectors of Projx * Projy 
and Projy * Projx, respectively, defined as follows: 
* Cx = (1 - `tau`) * X'DX + `tau` * Ix
* Cy = (1 - `tau`) * Y'DY + `tau` * Iy
* Cxy = X'DY 
* Projx = sqrt(D) * X * invCx * X' * sqrt(D)
* Projy = sqrt(D) * Y * invCx * Y' * sqrt(D)
where D is the observation (row) metric. Value `tau` = 0 can generate unstability when inverting the covariance 
matrices. Often, a better alternative is to use an epsilon value (e.g. `tau` = 1e-8) to get similar results as 
with pseudo-inverses.  

After normalized (and using uniform `weights`), the scores returned by the function are expected to be the same as 
those returned by functions `rcc` of the R packages `CCA` (González et al.) and `mixOmics` (Lê Cao et al.) whith their 
parameters lambda1 and lambda2 set to:
* lambda1 = lambda2 = `tau` / (1 - `tau`) * n / (n - 1)

See function `plscan` for the details on the `summary` outputs.

## References
González, I., Déjean, S., Martin, P.G.P., Baccini, A., 2008. CCA: An R Package to Extend Canonical
Correlation Analysis. Journal of Statistical Software 23, 1-14. https://doi.org/10.18637/jss.v023.i12

Hotelling, H. (1936): “Relations between two sets of variates”, Biometrika 28: pp. 321–377.

Lê Cao, K.-A., Rohart, F., Gonzalez, I., Dejean, S., Abadi, A.J., Gautier, B., Bartolo, F., Monget, P., 
Coquery, J., Yao, F., Liquet, B., 2022. mixOmics: Omics Data Integration Project. 
https://doi.org/10.18129/B9.bioc.mixOmics

Weenink, D. 2003. Canonical Correlation Analysis, Institute of Phonetic Sciences, Univ. of Amsterdam, 
Proceedings 25, 81-99.

## Examples
```julia
using Jchemo, JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "linnerud.jld2") 
@load db dat
@names dat
X = dat.X
Y = dat.Y
n, p = size(X)
q = nco(Y)

nlv = 3
bscal = :frob ; tau = 1e-8
model = cca(; nlv, bscal, tau)
fit!(model, X, Y)
@names model
@names model.fitm

@head model.fitm.Tx
@head transfbl(model, X, Y).Tx

@head model.fitm.Ty
@head transfbl(model, X, Y).Ty

res = summary(model, X, Y) ;
@names res
res.cortx2ty
res.rvx2tx
res.rvy2ty
res.rdx2tx
res.rdy2ty
res.corx2tx 
res.cory2ty 
```
"""
cca(; kwargs...) = JchemoModel(cca, nothing, kwargs)

function cca(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    n = nro(X)
    weights = mweight(ones(Q, n))
    cca(X, Y, weights; kwargs...)
end

function cca(X, Y, weights::Weight; kwargs...)
    cca!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function cca!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParCca, kwargs).par 
    @assert in([:none, :frob])(par.bscal) "Wrong value for argument 'bscal'."
    @assert 0 <= par.tau <= 1 "tau must be in [0, 1]"
    Q = eltype(X)
    p = nco(X)
    q = nco(Y)
    nlv = min(par.nlv, p, q)
    tau = convert(Q, par.tau) 
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
    sqrtw = sqrt.(weights.w)
    invsqrtw = 1 ./ sqrtw
    rweight!(X, sqrtw)
    rweight!(Y, sqrtw) 
    # End
    if tau == 0
        Cx = Symmetric(X' * X)
        Cy = Symmetric(Y' * Y)
    else
        Ix = Diagonal(ones(Q, p)) 
        Iy = Diagonal(ones(Q, q)) 
        if tau == 1
            Cx = Ix
            Cy = Iy
        else
            Cx = Symmetric((1 - tau) * X' * X + tau * Ix)
            Cy = Symmetric((1 - tau) * Y' * Y + tau * Iy)
        end
    end
    Cxy = X' * Y    
    Ux = cholesky(Hermitian(Cx)).U
    Uy = cholesky(Hermitian(Cy)).U
    invUx = inv(Ux)
    invUy = inv(Uy)
    A = invUx' * Cxy * invUy
    U, d, V = svd(A)
    Wx = invUx * U[:, 1:nlv]
    Wy = invUy * V[:, 1:nlv]
    d = d[1:nlv]
    Tx = rweight(X * Wx, invsqrtw) 
    Ty = rweight(Y * Wy, invsqrtw)
    Cca(Tx, Ty, Wx, Wy, d, bscales, xmeans, xscales, ymeans, yscales, weights, par)
end

""" 
    transfbl(object::Cca, X, Y; nlv = nothing)
Compute latent variables (LVs; = scores) from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which components (LVs) are computed.
* `Y` : Y-data for which components (LVs) are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transfbl(object::Cca, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = object.par.nlv
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    X = fcscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = fcscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    Tx = X * vcol(object.Wx, 1:nlv)
    Ty = Y * vcol(object.Wy, 1:nlv)
    (Tx = Tx, Ty)
end

"""
    summary(object::Cca, X, Y)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
* `Y` : The Y-data that was used to fit the model.
""" 
function Base.summary(object::Cca, X, Y)
    Q = eltype(X[1, 1])
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = fcscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = fcscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    ## To do: explvarx, explvary 
    #D = Diagonal(object.weights.w)   
    ## X
    #T = object.Tx 
    #sstot = frob(X, object.weights)^2
    #tt = diag(T' * D * X * X' * D * T) ./ diag(T' * D * T)
    ##tt = colsum(D * T .* T)  # = [1]
    #pvar =  tt / sstot
    #cumpvar = cumsum(pvar)
    #xvar = tt / n    
    #explvarx = DataFrame(nlv = 1:nlv, var = xvar, 
    #    pvar = pvar, cumpvar = cumpvar)
    explvarx = nothing 
    ## Y
    #T = object.Ty
    #sstot = frob2(Y, object.weights)
    #tt = diag(T' * D * Y * Y' * D * T) ./ diag(T' * D * T)
    ##tt = colsum(D * T .* T)  # = [1]
    #pvar =  tt / sstot
    #cumpvar = cumsum(pvar)
    #xvar = tt / n    
    #explvary = DataFrame(nlv = 1:nlv, var = xvar, 
    #    pvar = pvar, cumpvar = cumpvar)
    explvary = nothing
    ## Correlation between X- and Y-block LVs
    z = diag(corm(object.Tx, object.Ty, object.weights))
    cortx2ty = DataFrame(lv = 1:nlv, cor = z)
    ## RV(X, tx) and RV(Y, ty)
    nam = string.("lv", 1:nlv)
    z = zeros(Q, 1, nlv)
    for a = 1:nlv
        z[1, a] = rv(X, object.Tx[:, a], object.weights) 
    end
    rvx2tx = DataFrame(z, nam)
    for a = 1:nlv
        z[1, a] = rv(Y, object.Ty[:, a], object.weights) 
    end
    rvy2ty = DataFrame(z, nam)
    ## Redundancies (Average correlations) Rd(X, tx) and Rd(Y, ty)
    z[1, :] = rd(X, object.Tx, object.weights) 
    rdx2tx = DataFrame(z, nam)
    z[1, :] = rd(Y, object.Ty, object.weights) 
    rdy2ty = DataFrame(z, nam)
    ## Correlation between block variables and their block LVs
    z = corm(X, object.Tx, object.weights)
    corx2tx = DataFrame(z, string.("lv", 1:nlv))
    z = corm(Y, object.Ty, object.weights)
    cory2ty = DataFrame(z, string.("lv", 1:nlv))
    ## End
    (explvarx = explvarx, explvary, cortx2ty, rvx2tx, rvy2ty, rdx2tx, rdy2ty, corx2tx, cory2ty)
end

