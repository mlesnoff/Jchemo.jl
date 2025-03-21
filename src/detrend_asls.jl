"""
    detrend_asls(; kwargs...)
    detrend_asls(X; kwargs...)
Baseline correction of each row of X-data by asymmetric 
    least squares algorithm (ASLS).
* `X` : X-data (n, p).
Keyword arguments:
* `lb` : Penalizing (smoothness) parameter "lambda".
* `p` : Asymmetry parameter (0 < `p` << 1).
* `tol` : Tolerance value for stopping the iterations.  
* `maxit` : Maximum number of iterations.
* `verbose` : If `true`, nb. iterations are printed.

De-trend transformation: the function fits a baseline by ASLS (see Baek et al. 2015 section 2)
for each observation and returns the residuals (= signals corrected from the baseline).

Generally `0.001 ≤ p ≤ 0.1` is a good choice (for a signal with positive peaks) 
and `1e2 ≤ lb ≤ 1e9`, but exceptions may occur (Eilers & Boelens 2005).

## References

Baek, S.-J., Park, A., Ahn, Y.-J., Choo, J., 2015. Baseline correction using 
asymmetrically reweighted penalized least squares smoothing. Analyst 140, 250–257. 
https://doi.org/10.1039/C4AN01061B

Eilers, P. H., & Boelens, H. F. (2005). Baseline correction with asymmetric 
least squares smoothing. Leiden University Medical Centre Report, 1(1).

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
X = dat.X
year = dat.Y.year
s = year .<= 2012
Xtrain = X[s, :]
Xtest = rmrow(X, s)
wlst = names(dat.X)
wl = parse.(Float64, wlst)
plotsp(X, wl; nsamp = 20).f

## Example on 1 spectrum
i = 2
zX = Matrix(X)[i:i, :]
lb = 1e5 ; p = .001
model = detrend_asls(; lb, p)
fit!(model, zX)
zXc = transf(model, zX)   # = corrected spectrum 
B = zX - zXc              # = estimated baseline
f, ax = plotsp(zX, wl)
lines!(wl, vec(B); color = :blue)
lines!(wl, vec(zXc); color = :black)
f
```
""" 
detrend_asls(; kwargs...) = JchemoModel(detrend_asls, nothing, kwargs)

function detrend_asls(X; kwargs...)
    par = recovkw(ParDetrendAsls, kwargs).par
    DetrendAsls(par)
end

""" 
    transf(object::DetrendAsls, X)
    transf!(object::DetrendAsls, X)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::DetrendAsls, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::DetrendAsls, X::Matrix)
    n, zp = size(X)
    w = ones(zp) 
    z = similar(X, zp)
    z0 = copy(z)
    W = similar(X, zp, zp)
    D1 = sparse(diff(I(zp); dims = 1))
    D2 = diff(D1; dims = 1)
    C = D2' * D2
    lb = object.par.lb
    p = object.par.p
    tol = object.par.tol
    maxit = object.par.maxit
    verbose = object.par.verbose 
    verbose ? println("Nb. iterations:") : nothing
    @inbounds for i = 1:n
        iter = 1
        cont = true
        x = vrow(X, i)
        while cont
            z0 .= copy(z)
            W .= spdiagm(0 => w)    
            z .= cholesky!(Hermitian(W + lb * C)) \ (w .* x)
            ## Faster (but less safe) than:
            #z .= \(W + lb * C, w .* x)
            ## = (W + lb * C) \ (w .* x)    
            ## End 
            w .= p * (x .> z) + (1 - p) * (x .<= z)  
            dif = sum((z .- z0).^2)
            iter = iter + 1
            if (dif < tol) || (iter > maxit)
                cont = false
            end
        end
        verbose ? print(iter - 1, " ") : nothing
        X[i, :] .= x .- z
    end
end


