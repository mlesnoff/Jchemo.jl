## Not exported
## Translation of function 'spatial.median' available in the 
## script "PcaLocantore.R" of package rrcov v.1.4-3 on R CRAN 
## (Thanks to V. Todorov, 2016)
function colmedspa(X; delta = 1e-6) 
    X = ensure_mat(X)
    n, p = size(X)
    delta1 = delta * sqrt(p)
    mu0 = vec(median(X; dims = 1))
    X1 = similar(X)
    TT = similar(X)
    U = similar(X)
    w = similar(X, n)
    h = delta1 + 1
    while h > delta1
        TT .= mu0' .* ones(n, p)
        U .= (X - TT).^2
        w .= sqrt.(rowsum(U))
        w0 = median(w)
        ep = delta * w0
        s = w .<= ep 
        w[s] .= ep 
        s = w .> ep
        w[s] .= 1 ./ w[s]
        w ./= sum(w)
        for i = 1:n
            X1[i, :] .= w[i] * vrow(X, i)
        end
        mu = colsum(X1)
        h = sqrt(sum((mu - mu0).^2))
        mu0 .= copy(mu)
    end
    mu0
end


