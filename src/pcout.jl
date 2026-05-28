"""
    pcout(X, V; scal = false)
    pcout!(X::Matrix, V::Matrix; scal = false)
Pcout algorihm.
* `X` : X-data (n, p).
* `V` : A projection matrix (p, nlv) representing the directions of the projection pursuit.
Keyword arguments:
* `` : .

## References

Filzmoser, P., Maronna, R., Werner, M., 2008. Outlier identification in high dimensions. 
Computational Statistics & Data Analysis 52, 1694–1711. https://doi.org/10.1016/j.csda.2007.05.018

Filzmoser, P. 2026. mvoutlier: Multivariate Outlier Detection Based on Robust Methods. 
R package version 2.1.4. https://cran.r-project.org/package=mvoutlier

## Examples
```julia
```
""" 
function pcout(X)
    pcout!(copy(ensure_mat(X))) 
end

function pcout!(X::Matrix{Q}; explvar::Q = .99, critM1 = 1 / 3, critc1 = 2.5, critM2 = 1 / 4, critc2 = 0.99, 
    cs = .25, outbound = 0.25) where Q <: AbstractFloat
    n = nro(X)
    d = similar(X, n)
    w1 = similar(d)
    w2 = similar(d)
    wfinal = similar(d)
    wfinal01 = similar(d)
    fcscale!(X, colmed(X), colmad(X))
    res = svd(fcenter(X, colmean(X))) ;
    sv = res.S.^2 / (n - 1)
    nlv = findall(cumsum(sv) / sum(sv) .> explvar)[1]
    distr = Chisq(nlv)    
    q = quantile(distr, .5)
    T = X * vcol(res.V, 1:nlv)             # PCs
    fcscale!(T, colmed(T), colmad(T))      # centered and scaled PCs (by median and mad)
    # Phase 1
    w = abs.(colmean(T.^4) .- 3)           # robust kurtosis for each PC
    w ./= sum(w)
    Tw = fweightc(T, w)                    # weighted PCs
    d .= sqrt.(rowsum(Tw.^2))              # weighted Mahalanobis distance in the PC space (RDi in Eq12)
    dist1 = d * sqrt(q) / median(d)        # di = transformed RDi
    M1 = quantile(dist1, critM1)  
    const1 = medv(dist1) + critc1 * madv(dist1)
    @inbounds for i in eachindex(w1)
        if dist1[i] <= M1
            w1[i] = 1
        elseif dist1[i] >= const1
            w1[i] = 0
        else
            w1[i] = (1 - ((dist1[i] - M1) / (const1 - M1))^2)^2
        end
    end 
    ## Phase 2
    d .= sqrt.(rowsum(T.^2))
    dist2 = d * sqrt(q) / median(d)
    M2 = sqrt(quantile(distr, critM2))
    const2 = sqrt(quantile(distr, critc2)) 
    @inbounds for i in eachindex(w2)
        if dist2[i] <= M2
            w2[i] = 1
        elseif dist2[i] >= const2
            w2[i] = 0
        else
            w2[i] = (1 - ((dist2[i] - M2) / (const2 - M2))^2)^2
        end
    end 
    ## End
    @. wfinal = ((w1 + cs) * (w2 + cs)) / (1 + cs)^2
    @. wfinal01 = round(wfinal + 0.5 - outbound)
    (wfinal01 = wfinal01, wfinal, wloc = w1, wscat = w2, dist1, dist2, M1, const1, M2, const2)
end






