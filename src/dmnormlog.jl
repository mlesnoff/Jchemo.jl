struct Dmnormlog
    mu
    Uinv 
    logdetS
    logcst
end

function dmnormlog(X = nothing; mu = nothing, S = nothing,
        simpl = false)
    isnothing(S) ? zS = nothing : zS = copy(S)
    dmnormlog!(X; mu = mu, S = zS, simpl = simpl)
end

function dmnormlog!(X = nothing; mu = nothing, S = nothing,
        simpl = false)
    !isnothing(X) ? X = ensure_mat(X) : nothing
    if isnothing(mu)
        mu = vec(mean(X, dims = 1))
    end
    if isnothing(S)
        S = cov(X; corrected = true)
    end
    if simpl 
        logcst = 0
        logdetS = 0
    else
        p = nro(S)
        logcst = -p / 2 * log(2 * pi)
        logdetS = logdet(S)
    end  
    U = cholesky!(Hermitian(S)).U
    LinearAlgebra.inv!(U)
    Dmnormlog(mu, U, logdetS, logcst)
end

function predict(object::Dmnormlog, X)
    X = ensure_mat(X)
    mu = reshape(object.mu, 1, length(object.mu))
    d = mahsqchol(X, mu, object.Uinv)
    @. d = object.logcst - object.logdetS / 2 - d / 2
    (pred = d,)
end


    