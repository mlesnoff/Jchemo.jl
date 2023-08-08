# Mangamana et al 2021, section 2.1.2
# wx is optimized, instead of wy in plsmang
function splsnipals(X, Y; nlv,
        nvar = nco(X),
        tol = 1e-8, maxit = 100)
    X = copy(ensure_mat(X))
    Y = copy(ensure_mat(Y))
    n, p = size(X)
    q = nco(Y)
    nlv = min(nlv, n, p)
    length(nvar) == 1 ? nvar = repeat([nvar], nlv) : nothing
    xmeans = colmean(X) 
    ymeans = colmean(Y)   
    center!(X, xmeans)
    center!(Y, ymeans)
    Tx  = similar(X, n, nlv)
    Ty = copy(Tx)
    Wx  = similar(X, p, nlv)
    Px  = copy(Wx)
    Wy  = similar(X, q, nlv)
    Wytild  = copy(Wy)
    Py = similar(X, n, n)
    TTx = similar(X, nlv)
    tx  = similar(X, n)
    ty  = copy(tx)
    wx = similar(X, p)
    absw = copy(wx)
    px  = copy(wx)
    wy  = similar(X, q)
    wytild = copy(wy)
    lambda = copy(TTx)
    covtot = copy(TTx)
    iter = Int64.(ones(nlv))
    @inbounds for a = 1:nlv
        cont = true
        wx .= rand(p)
        wx ./= norm(wx)
        Py .= Y * Y'
        nrm = p - nvar[a]
        while cont
            w0 = copy(wx)
            tx .= X * wx
            ty .= Py * tx / norm(Y' * tx)
            wx .= X' * ty
            ## Sparsity
            if nrm > 0
                absw .= abs.(wx)
                sel = sortperm(absw; rev = true)[1:nvar[a]]
                wmax = wx[sel]
                wx .= zeros(p)
                wx[sel] .= wmax
                delta = maximum(sort(absw)[1:nrm])
                wx .= soft.(wx, delta)
            end
            ## End
            wx ./= norm(wx)
            dif = sum((wx .- w0).^2)
            iter[a] = iter[a] + 1
            if (dif < tol) || (iter[a] > maxit)
                cont = false
            end
        end
        lambda[a] = tx' * Py * tx
        covtot[a] = tr(X' * Py * X)      
        tt = dot(tx, tx)
        mul!(px, X', tx)
        px ./= tt
        mul!(wytild, Y', tx)
        wytild ./= tt
        wy .= wytild / norm(wytild)
        # Deflation
        X .-= tx * px'
        Y .-= tx * wytild'
        # End
        Tx[:, a] .= tx 
        Ty[:, a] .= ty         
        Wx[:, a] .= wx
        Px[:, a] .= px  
        Wy[:, a] .= wy
        Wytild[:, a] .= wytild
        TTx[a] = tt
     end
     #R = Wx * inv(Px' * Wx)
     (Tx = Tx, Ty, Wx, Px, Wy, Wytild, TTx, lambda, covtot, 
         xmeans, ymeans, weights, iter)
end

