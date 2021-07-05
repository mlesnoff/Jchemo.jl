struct Cglsr
    B::Matrix{Float64}
    g::Vector{Float64}
    xmeans::Vector{Float64}
    ymeans::Vector{Float64}
    F::Union{Array{Float64}, Nothing}
end

"""
    cglsr(X, y; nlv, reorth = true, filt = false)
Conjugate gradient algorithm for the normal equations (CGLS) (Björck 1996).
* `X` : matrix (n, p), or vector (n,).
* `y` : matrix (n, q), or vector (n,).
* `nlv` : Nb. iterations.
* `reorth` : If `true`, a Gram-Schmidt reorthogonalization of the normal equation 
    residual vectors is done.
* `filt` : If `true`, the filter factors are computed (output `F`).

CGLS algorithm 7.4.1 Bjorck 1996, p.289

The code for re-orthogonalization, and filter factors computations (Vogel 1987, Hansen 1998), 
is a transcription (with few adaptations) of the matlab function `cgls` 
(Saunders et al. https://web.stanford.edu/group/SOL/software/cgls/; Hansen 2008).

`X` and `y` are internally centered. 

The in-place version modifies externally `X` and `y`. 

## References

Björck, A., 1996. Numerical Methods for Least Squares Problems, Other Titles in Applied Mathematics. 
Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9781611971484

Hansen, P.C., 1998. Rank-Deficient and Discrete Ill-Posed Problems, Mathematical Modeling and Computation. 
Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898719697

Hansen, P.C., 2008. Regularization Tools version 4.0 for Matlab 7.3. 
Numer Algor 46, 189–194. https://doi.org/10.1007/s11075-007-9136-9

Manne R. Analysis of two partial-least-squares algorithms for multivariate calibration. Chemometrics Intell.
Lab. Syst. 1987; 2: 187–197.

Phatak A, De Hoog F. Exploiting the connection between
PLS, Lanczos methods and conjugate gradients: alternative proofs of some properties of PLS. J. Chemometrics
2002; 16: 361–367.

Vogel, C. R.,  "Solving ill-conditioned linear systems using the conjugate gradient method", 
Report, Dept. of Mathematical Sciences, Montana State University, 1987.
""" 
function cglsr(X, y; nlv, reorth = true, filt = false)
    cglsr!(copy(X), copy(y); nlv = nlv, reorth = reorth, filt = filt)
end
function cglsr!(X, y; nlv, reorth = true, filt = false)
    X = ensure_mat(X)
    n = size(X, 1)
    p = size(X, 2)
    xmeans = colmeans(X) 
    ymeans = colmeans(y)   
    center!(X, xmeans)
    center!(y, ymeans)
    # Pre-allocation and initialization
    B = similar(X, p, nlv)
    b = zeros(p) 
    r = y       # r = y - X * b, with b = 0
    s = X' * r
    zp = copy(s)
    q = similar(X, n)
    g = dot(s, s)
    gnew = similar(X, nlv)
    if(reorth)
        A = similar(X, p, nlv + 1)
        A[:, 1] .= s ./ sqrt(g)
    end
    F = nothing
    if(filt)
        eig = svd(X).S.^2
        if(n < p)
            eig = [eig ; zeros(0, p - n)]
        end
        F = similar(X, p, nlv) 
        Fd = similar(X, p) 
    end
    fudge_thr = 1e-4
    # End
    for j in 1:nlv
        mul!(q, X, zp)
        alpha = g / dot(q, q)
        b .+= alpha * zp 
        r .-= alpha * q 
        mul!(s, X', r)
        # Reorthogonalize s to previous s-vectors
        if reorth
            for i in 1:j
                v = vcol(A, i)
                s .-= dot(v, s) * v
            end
            A[:, j + 1] .= s ./ sqrt(dot(s, s))
        end
        # End
        gnew[j] = dot(s, s) 
        beta = gnew[j] / g
        g = copy(gnew[j])
        zp .= s .+ beta * zp
        B[:, j] .= b
        # Compute filter factors
        if filt
            if j == 1
                F[:, 1] .= alpha * eig
                Fd .= eig .- eig .* F[:, 1] .+ beta * eig
            else
                F[:, j] .= F[:, j - 1] .+ alpha * Fd
                Fd .= eig - eig .* F[:, j] .+ beta * Fd
            end
            if j > 2
                u = (abs.(F[:, j - 1] .- 1) .< fudge_thr) .* (abs.(F[:, j - 2] .- 1) .< fudge_thr)
                u = findall(u)
                if length(u) > 0
                    F[u, j] .= ones(length(u))
                end
            end
        end 
        # End
    end
    Cglsr(B, gnew, xmeans, ymeans, F)
end

"""
    coef(object::Cglsr)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `nlv` : Nb. iterations to consider. If nothing, it is the maximum nb. iterations.
""" 
function coef(object::Cglsr; nlv = nothing)
    a = size(object.B, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    B = object.B[:, nlv:nlv]
    int = object.ymeans' .- object.xmeans' * B
    (int = int, B = B)
end

"""
    predict(object::Cglsr, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The maximal fitted model.
* `X` : Matrix (m, p) for which predictions are computed.
* `nlv` : Nb. iterations, or collection of nb. iterations, to consider. 
If nothing, it is the maximum nb. iterations.
""" 
function predict(object::Cglsr, X; nlv = nothing)
    a = size(object.B, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(le_nlv)
    @inbounds for i = 1:le_nlv
        z = coef(object; nlv = nlv[i])
        pred[i] = z.int .+ X * z.B
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end



