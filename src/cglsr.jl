"""
    cglsr(X, y; nlv, gs = true, filt = false, scal::Bool = false)
    cglsr!(X::Matrix, y::Matrix; nlv, gs = true, filt = false, scal::Bool = false)
Conjugate gradient algorithm for the normal equations (CGLS; Björck 1996).
* `X` : X-data  (n, p).
* `y` : Univariate Y-data (n).
* `nlv` : Nb. CG iterations.
* `gs` : If `true`, a Gram-Schmidt gsogonalization of the normal equation 
    residual vectors is done.
* `filt` : Logical indicating if the CG filter factors are computed (output `F`).
* `scal` : Boolean. If `true`, each column of `X` and `y` 
    is scaled by its uncorrected standard deviation.

`X` and `y` are internally centered. 

CGLS algorithm "7.4.1" Bjorck 1996, p.289. The part of the code computing the 
re-orthogonalization (Hansen 1998) and filter factors (Vogel 1987, Hansen 1998) 
is a transcription (with few adaptations) of the Matlab function `cgls` 
(Saunders et al. https://web.stanford.edu/group/SOL/software/cgls/; Hansen 2008).

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

nlv = 12 ;
fm = cglsr(Xtrain, ytrain; nlv = nlv) ;

zcoef = Jchemo.coef(fm)
zcoef.int
zcoef.B
Jchemo.coef(fm; nlv = 7).B

res = Jchemo.predict(fm, Xtest) ;
res.pred
rmsep(ytest, res.pred)
plotxy(pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f    
```
""" 
cglsr(X, y; kwargs...) = cglsr!(copy(ensure_mat(X)), 
    copy(ensure_mat(y)); values(kwargs)...)

function cglsr!(X::Matrix, y::Matrix; kwargs...)
    Q = eltype(X)   
    n, p = size(X)
    q = nco(y)
    nlv = min(n, p, par.nlv)
    xmeans = colmean(X)
    ymeans = colmean(y)
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal 
        xscales .= colstd(X)
        yscales .= colstd(y)
        fcscale!(X, xmeans, xscales)
        fcscale!(y, ymeans, yscales)
    else
        fcenter!(X, xmeans)
        fcenter!(y, ymeans)
    end
    # Pre-allocation and initialization
    B = similar(X, p, nlv)
    b = zeros(Q, p) 
    r = vec(y)       # r = y - X * b, with b = 0
    s = X' * r
    zp = copy(s)
    q = similar(X, n)
    g = dot(s, s)
    gnew = similar(X, nlv)
    if par.gs 
        A = similar(X, p, nlv + 1)
        A[:, 1] .= s ./ sqrt(g)
    end
    F = nothing
    if par.filt
        eig = svd(X).S.^2
        if(n < p)
            eig = [eig ; zeros(p - n)]
        end
        F = similar(X, p, nlv) 
        Fd = similar(X, p) 
    end
    fudge = 1e-4
    # End
    @inbounds for j in 1:nlv
        mul!(q, X, zp)
        alpha = g / dot(q, q)
        b .+= alpha * zp 
        r .-= alpha * q 
        mul!(s, X', r)
        # Reorthogonalize s to previous s-vectors
        if par.gs
            @inbounds for i in 1:j
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
        # Filter factors
        # fudge threshold is used to prevent filter factors from exploding
        if par.filt
            if j == 1
                F[:, 1] .= alpha * eig
                Fd .= eig .- eig .* F[:, 1] .+ beta * eig
            else
                F[:, j] .= F[:, j - 1] .+ alpha * Fd
                Fd .= eig - eig .* F[:, j] .+ beta * Fd
            end
            if j > 2
                u = (abs.(F[:, j - 1] .- 1) .< fudge) .* (abs.(F[:, j - 2] .- 1) .< fudge)
                u = findall(u)
                if length(u) > 0
                    F[u, j] .= ones(Q, length(u))
                end
            end
        end 
        # End
    end
    Cglsr(B, gnew, xmeans, xscales, ymeans, yscales, F)
end

"""
    coef(object::Cglsr)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
""" 
function coef(object::Cglsr; nlv = nothing)
    a = nco(object.B)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    W = Diagonal(object.yscales)
    B = Diagonal(1 ./ object.xscales) * object.B[:, nlv:nlv] *  W
    int = object.ymeans' .- object.xmeans' * B
    (B = B, int = int)
end

"""
    predict(object::Cglsr, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. iterations, or collection of nb. iterations, to consider. 
If nothing, it is the maximum nb. iterations.
""" 
function predict(object::Cglsr, X; nlv = nothing)
    X = ensure_mat(X)
    a = nco(object.B)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(le_nlv, Matrix{eltype(X)})
    @inbounds for i = 1:le_nlv
        z = coef(object; nlv = nlv[i])
        pred[i] = z.int .+ X * z.B
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end



