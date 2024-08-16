"""
    asls(X; kwargs...)
Baseline correction of each row of X-data by asymmetric 
    least squares algorithm (ASLS).
* `X` : X-data (n, p).
Keyword arguments:
* `lb` : Penalizing (smoothing) parameter "lambda".
* `p` : Asymmetry parameter (0 < `p` << 1).
* `maxit` : Maximum number of iterations.
* `verbose` : If `true`, nb. iterations are printed.

See Andries & Nikzad-Langerodi 2024 Section 2.2.

## References

Andries, E., Nikzad-Langerodi, R., 2024. Supervised and Penalized Baseline 
Correction. https://doi.org/10.48550/arXiv.2310.18306

Eilers, P. H., & Boelens, H. F. (2005). Baseline correction with asymmetric 
least squares smoothing. Leiden University Medical Centre Report, 1(1), p 5

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
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
lb = 200 ; p = .001
mod = model(asls; lb, p)
fit!(mod, zX)
zXc = transf(mod, zX)   # = corrected spectrum 
B = zX - zXc            # = estimated baseline
f, ax = plotsp(zX, wl)
lines!(wl, vec(B); color = :blue)
lines!(wl, vec(zXc); color = :black)
f
```
""" 
function asls(X; kwargs...)
    par = recovkw(ParAsls, kwargs).par
    Asls(par)
end

""" 
    transf(object::Asls, X)
    transf!(object::Asls, X)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Asls, X)
    X = copy(ensure_mat(X))
    transf!(object, X)
    X
end

function transf!(object::Asls, X::Matrix)
    n, zp = size(X)
    w = ones(zp) 
    z = similar(X, zp)
    z0 = copy(z)
    W = similar(X, zp, zp)
    D1 = sparse(diff(I(zp); dims = 1))
    D2 = diff(D1; dims = 1)
    C = D2' * D2
    lb2 = object.par.lb^2
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
            z .= \(W + lb2 * C, w .* x)
            ## same as:
            #z .= (W + lb2 * C) \ (w .* x)    
            ## End 
            w .= p * (x .> z) + (1 - p) * (x .< z)  
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


