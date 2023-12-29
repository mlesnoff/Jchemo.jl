"""
    rrr(; kwargs...)
    rrr(X, Y; kwargs...)
    rrr(X, Y, weights::Weight; 
        kwargs...)
    rr!(X::Matrix, Y::Matrix, weights::Weight; 
        kwargs...)
Reduced rank regression (RRR, *aka* RA).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `tau` : Regularization parameter (∊ [0, 1]).
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.
 
Reduced rank regression, also referred to as redundancy 
analysis (RA) regression. In this function, the RA uses 
the Nipals algorithm presented in Mangamana et al 2021, 
section 2.1.1.

A continuum regularization is available. After block centering 
and scaling, the covariances matrices are computed as follows: 
* Cx = (1 - `tau`) * X'DX + `tau` * Ix
where D is the observation (row) metric. Value `tau` = 0 
can generate unstability when inverting the covariance matrices. 
A better alternative is generally to use an epsilon value 
(e.g. `tau` = 1e-8) to get similar results as with pseudo-inverses.  

## References
Bougeard, S., Qannari, E.M., Lupo, C., Chauvin, C., 2011. 
Multiblock redundancy analysis from a user’s perspective. 
Application in veterinary epidemiology. Electronic Journal of 
Applied Statistical Analysis 4, 203-214–214. 
https://doi.org/10.1285/i20705948v4n2p203

Bougeard, S., Qannari, E.M., Rose, N., 2011. Multiblock redundancy 
analysis: interpretation tools and application in epidemiology. 
Journal of Chemometrics 25, 467–475. https://doi.org/10.1002/cem.1392 

Tchandao Mangamana, E., Glèlè Kakaï, R., Qannari, E.M., 2021. 
A general strategy for setting up supervised methods of multiblock 
data analysis. Chemometrics and Intelligent Laboratory Systems 
217, 104388. 
https://doi.org/10.1016/j.chemolab.2021.104388

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 1
tau = 1e-4
mod = rrr(; nlv, tau) ;
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)
@head mod.fm.T

coef(mod)
coef(mod; nlv = 3)

@head transf(mod, Xtest)
@head transf(mod, Xtest; nlv = 3)

res = predict(mod, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f   
```
"""
function rrr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    rrr(X, Y, weights; kwargs...)
end

function rrr(X, Y, weights::Weight; kwargs...)
    rrr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; kwargs...)
end

function rrr!(X::Matrix, Y::Matrix, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert 0 <= par.tau <=1 "tau must be in [0, 1]"
    Q = eltype(X)
    n, p = size(X)
    q = nco(Y)
    nlv = min(par.nlv, p, q)
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
    # Row metric
    X .= sqrtw .* X
    Y .= sqrtw .* Y
    # Pre-allocation
    Tx  = similar(X, n, nlv)
    Wx  = similar(X, p, nlv)
    Wy  = similar(X, q, nlv)
    Wytild  = copy(Wy)
    Px  = similar(X, p, nlv)
    Cx = similar(X, p, p)
    invCx = copy(Cx)
    Ix = Diagonal(ones(Q, p))
    TTx = similar(X, nlv)
    tx  = similar(X, n)
    ty  = copy(tx)
    wx  = similar(X, p)
    wy  = similar(X, q)
    wytild = copy(wy)
    px = similar(X, p)
    lambda = copy(TTx)
    covtot = copy(TTx)
    niter = Int.(ones(nlv))
    tau = par.tau
    @inbounds for a = 1:nlv
        cont = true
        iter = 1
        wy .= ones(Q, q)
        wy ./= norm(q)
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
        while cont
            w0 = copy(wy)
            ty .= Y * wy
            tx .= X * invCx * X' * ty
            wy .= Y' * tx
            wy ./= norm(wy)
            dif = sum((wy .- w0).^2)
            iter = iter + 1
            if (dif < par.tol) || (iter[a] > par.maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        lambda[a] = ty' * X * invCx * X' * ty
        covtot[a] = tr(Y' * X * invCx * X' * Y)
        ttx = dot(tx, tx)
        mul!(px, X', tx)
        px ./= ttx
        mul!(wytild, Y', tx)
        wytild ./= ttx
        # For Rx
        tty = dot(ty, ty)
        wx .= invCx * X' * ty / tty
        wx .= wx / norm(wx)
        # Deflation
        X .-= tx * px'
        Y .-= tx * wytild'
        # End
        Tx[:, a] .= tx   
        Px[:, a] .= px
        Wx[:, a] .= wx
        Wytild[:, a] .= wytild
        TTx[a] = ttx
     end
     Rx = Wx * inv(Px' * Wx)
     Tx .= (1 ./ sqrtw) .* Tx
     Plsr(Tx, Px, Rx, Wx, Wytild, TTx, 
         xmeans, xscales, ymeans, yscales, weights, niter,
         kwargs, par)
end


