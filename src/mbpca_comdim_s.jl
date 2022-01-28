struct MbpcaComdim3
    T::Array{Float64} 
    U::Array{Float64}
    W::Array{Float64}
    Tb::Vector{Array{Float64}}
    W_bl::Vector{Array{Float64}}
    lb::Array{Float64}
    mu::Vector{Float64}
    xmeans::Vector{Vector{Float64}}
    scal::Vector{Float64}
    weights::Vector{Float64}
    niter::Vector{Float64}
end

"""
    mbpca_comdim_s(X_bl, weights = ones(size(X_bl[1], 1)); nlv,
        bscaling = "none", tol = sqrt(eps(1.)), maxit = 200)
Common Components and Specific Weights Analysis (CCSWA): Multiblock ComDim PCA.
* `X_bl` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `weights` : Weights of the observations (rows). 
* `nlv` : Nb. latent variables (LVs) to compute.
* `bscaling` : Type of block scaling (`"none"`, `"frob"`). 
    See functions `blockscal`.
* `tol` : Tolerance value for convergence.
* `niter` : Maximum number of iterations.

"SVD" algorithm of Hannafi & Qannari 2008 p.84.

Vector `weights` is internally normalized to sum to 1.

The function returns several objects, in particular:
* `T` : The non normed global scores.
* `U` : The normed global scores.
* `W` : The global loadings.
* `Tb` : The block scores.
* `W_bl` : The block loadings.
* `lb` : The specific weights (saliences) "lambda".
* `mu` : The sum of the squared saliences.

Function `summary` returns: 
* `explvarx` : Proportion of the X total inertia (sum of the squared norms of the 
    blocks X_k) explained by each global score.
* `explvarxx` : Proportion of the XX' total inertia (sum of the squared norms of the
    products X_k * X_k') explained by each global score 
    (= indicator "V" in Qannari et al. 2000, Hanafi et al. 2008).
* `sal2` : Proportion of the squared saliences (specific weights)
    of each block within each global score. 
* `contr_block` : Contribution of each block to the global scores 
    (= proportions of the saliences "lambda" within each score)
* `explX` : Proportion of the block X inertia explained by each global score.
* `cort2x` : Correlation between the global scores and the original variables.  
* `cort2tb` : Correlation between the global scores and the block scores.

## References
Cariou, V., Qannari, E.M., Rutledge, D.N., Vigneau, E., 2018. ComDim: From multiblock data 
analysis to path modeling. Food Quality and Preference, Sensometrics 2016: 
Sensometrics-by-the-Sea 67, 27–34. https://doi.org/10.1016/j.foodqual.2017.02.012

Cariou, V., Jouan-Rimbaud Bouveresse, D., Qannari, E.M., Rutledge, D.N., 2019. 
Chapter 7 - ComDim Methods for the Analysis of Multiblock Data in a Data Fusion 
Perspective, in: Cocchi, M. (Ed.), Data Handling in Science and Technology, 
Data Fusion Methodology and Applications. Elsevier, pp. 179–204. 
https://doi.org/10.1016/B978-0-444-63984-4.00007-7

Ghaziri, A.E., Cariou, V., Rutledge, D.N., Qannari, E.M., 2016. Analysis of multiblock 
datasets using ComDim: Overview and extension to the analysis of (K + 1) datasets. 
Journal of Chemometrics 30, 420–429. https://doi.org/10.1002/cem.2810

Hanafi, M., 2008. Nouvelles propriétés de l’analyse en composantes communes et 
poids spécifiques. Journal de la société française de statistique 149, 75–97.

Qannari, E.M., Wakeling, I., Courcoux, P., MacFie, H.J.H., 2000. Defining the underlying 
sensory dimensions. Food Quality and Preference 11, 151–154. 
https://doi.org/10.1016/S0950-3293(99)00069-5
"""
function mbpca_comdim_s(X_bl, weights = ones(size(X_bl[1], 1)); nlv,
        bscaling = "none", tol = sqrt(eps(1.)), maxit = 200)
    nbl = length(X_bl)
    X = copy(X_bl)
    n = size(X[1], 1)
    weights = mweights(weights)
    sqrtw = sqrt.(weights)
    sqrtD = Diagonal(sqrtw)
    xmeans = list(nbl, Vector{Float64})
    p = fill(0, nbl)
    @inbounds for k = 1:nbl
        p[k] = size(X[k], 2)
        xmeans[k] = colmeans(X[k], weights)   
        X[k] = center(X[k], xmeans[k])
    end
    bscaling == "none" ? scal = ones(nbl) : nothing
    if bscaling == "frob"
        res = blockscal_frob(X, weights) 
        scal = res.scal
        X = res.X
    end
    @inbounds for k = 1:nbl
        X[k] .= sqrtD * X[k]
    end
    # Pre-allocation
    u = similar(X[1], n)
    U = similar(X[1], n, nlv)
    tb = copy(u)
    Tb = list(nlv, Matrix{Float64})
    for a = 1:nlv ; Tb[a] = similar(X[1], n, nbl) ; end
    W_bl = list(nbl, Matrix{Float64})
    for k = 1:nbl ; W_bl[k] = similar(X[1], p[k], nlv) ; end
    lb = similar(X[1], nbl, nlv)
    sv = similar(X[1], nlv)
    T_B = similar(X[1], n, nbl)
    W = similar(X[1], nbl, nlv)
    niter = zeros(nlv)
    # End
    @inbounds for a = 1:nlv
        zX = reduce(hcat, X)
        u .= nipals(zX).u
        iter = 1
        cont = true
        while cont
            u0 = copy(u)
            for k = 1:nbl
                w = X[k]' * u 
                w ./= norm(w)
                mul!(tb, X[k], w) 
                alpha = abs.(dot(tb, u))
                T_B[:, k] = alpha * tb
                lb[k, a] = alpha^2
                Tb[a][:, k] .= tb
                W_bl[k][:, a] .= w
            end
            res = nipals(T_B)
            u .= res.u
            dif = sum((u - u0).^2)
            iter = iter + 1
            if (dif < tol) || (iter > maxit)
                cont = false
            end
        end
        niter[a] = iter - 1
        U[:, a] .= u
        W[:, a] .= res.v
        sv[a] = res.sv       # sv = sqrt(mu)
        @inbounds for k = 1:nbl
            X[k] .-= u * (u' * X[k])
            # Same as:
            #Px = sqrt(lb[k, a]) * W_bl[k][:, a]'
            #X[k] .-= u * Px
        end
    end
    mu = colsums(lb.^2)
    T = Diagonal(1 ./ sqrtw) * (sqrt.(mu)' .* U)
    MbpcaComdim3(T, U, W, Tb, W_bl, lb, mu, 
        xmeans, scal, weights, niter)
end

function transform(object::MbpcaComdim3, X_bl; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    nbl = length(X_bl)
    X = copy(X_bl)
    m = size(X[1], 1)
    @inbounds for k = 1:nbl
        X[k] = center(X[k], object.xmeans[k])
    end
    X = blockscal(X; scal = object.scal).X
    U = similar(X[1], m, nlv)
    T_B = similar(X[1], m, nbl)
    u = similar(X[1], m)
    for a = 1:nlv
        z = zeros(nbl)
        for k = 1:nbl
            T_B[:, k] .= X[k] * object.W_bl[k][:, a]
        end
        T_B .= sqrt.(object.lb[:, a])' .* T_B
        u .= (T_B / sqrt(object.mu[a])) * object.W[:, a]
        U[:, a] .= u
        @inbounds for k = 1:nbl
            Px = sqrt(object.lb[k, a]) * object.W_bl[k][:, a]'
            X[k] .-= u * Px
        end
    end
    sqrt.(object.mu)' .* U # = T
end

function summary(object::MbpcaComdim3, X_bl)
    nbl = length(X_bl)
    nlv = size(object.T, 2)
    X = copy(X_bl)
    n = size(X[1], 1)
    sqrtw = sqrt.(object.weights)
    sqrtD = Diagonal(sqrtw)
    @inbounds for k = 1:nbl
        X[k] = center(X[k], object.xmeans[k])
    end
    X = blockscal(X; scal = object.scal).X
    @inbounds for k = 1:nbl
        X[k] .= sqrtD * X[k]
    end
    # Explained_X
    sstot = zeros(nbl)
    @inbounds for k = 1:nbl
        sstot[k] = sum(colnorms2(X[k]))
    end
    tt = colsums(object.lb)    
    pvar = tt / sum(sstot)
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(pc = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    # Explained_XXt (indicator "V")
    S = list(nbl, Matrix{Float64})
    sstot_xx = 0 
    @inbounds for k = 1:nbl
        S[k] = X[k] * X[k]'
        sstot_xx += sum(colnorms2(S[k]))
    end
    tt = object.mu
    pvar = tt / sstot_xx
    cumpvar = cumsum(pvar)
    explvarxx = DataFrame(pc = 1:nlv, var = tt, pvar = pvar, cumpvar = cumpvar)
    # Prop saliences^2
    sal2 = copy(object.lb)
    for a = 1:nlv
        sal2[:, a] .= object.lb[:, a].^2 / object.mu[a]
    end
    sal2 = DataFrame(sal2, string.("pc", 1:nlv))
    # Contribution of the blocks to superscores = lb proportions (contrib)
    z = scale(object.lb, colsums(object.lb))
    contr_block = DataFrame(z, string.("pc", 1:nlv))
    # Proportion of inertia explained for each block (explained.X)
    # = object.lb if bscaling = "frob" 
    z = scale((object.lb)', sstot)'
    explX = DataFrame(z, string.("pc", 1:nlv))
    # Correlation between the superscores and the original variables (globalcor)
    zX = reduce(hcat, X)
    z = cor(zX, object.U)  
    cort2x = DataFrame(z, string.("pc", 1:nlv))  
    # Correlation between the superscores and the block_scores (cor.g.b)
    z = list(nlv, Matrix{Float64})
    @inbounds for a = 1:nlv
        z[a] = cor(object.Tb[a], object.U[:, a])
    end
    cort2tb = DataFrame(reduce(hcat, z), string.("pc", 1:nlv))
    #zrv = list(nlv, Float64)
    #@inbounds for k = 1:nbl
    #    zrv[k] = rv(S[k], object.T * object.T')
    #end
    (explvarx = explvarx, explvarxx, sal2, contr_block, explX, 
        cort2x, cort2tb, rv = nothing)
end


