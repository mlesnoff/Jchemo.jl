function decompx(X, f::StatsModels.FormulaTerm, dat::DataFrame)
    X = ensure_mat(X)
    Q = eltype(X)
    n = nro(X)
    ymeans = colmean(X)
    Xc = fcenter(X, ymeans)
    Xm = ones(Q, n) * ymeans'
    ## Contrasts
    contr = EffectsCoding()   # sum-to-zero
    term_princ = Symbol.(terms(f.rhs))     # [2:end]
    nterm_princ = length(term_princ)
    tupl = (; zip(term_princ, repeat([contr], nterm_princ))...)
    nam = @names tupl
    contrasts = Dict{Symbol, EffectsCoding}()
    for i in term_princ
        contrasts[i] = contr
    end 
    ## D, B
    mf = ModelFrame(f, dat; contrasts)
    fs = apply_schema(f, mf.schema)
    resp, D = modelcols(fs, dat) ;
    dfm = nco(D) + 1 # include intercept
    dfr = n - dfm
    B = inv(D' * D) * D' * Xc
    ## Assign terms
    term_rhs = fs.rhs.terms
    nterm_rhs = length(term_rhs) 
    assign = StatsModels.asgn(term_rhs)
    #AnovaBase.dof_asgn(assign)
    ## Fit (including Intercept term) and E
    namfit = vcat("Intercept", collect(string.(term_rhs)))
    fit = list(Matrix{Q}, nterm_rhs + 1)
    fit[1] = copy(Xm)
    c = zeros(dfm - 1)
    for i in eachindex(term_rhs)
        c[assign .== i] .= 1
        C = diagm(c)
        M = D * C
        fit[i + 1] = M * B
        c .= zeros(dfm - 1)
    end
    dffit = vcat(1, tab(assign).vals)
    E = Xc - D * B
    ss = (sst = frob2(X), ssfit =  frob2.(fit), ssr = frob2(E))
    df = (dffit = dffit, dfr, n)
    ## 'fit' and 'namfit' could be replaced by a named tuple 'fit'
    (fit = fit, namfit, E, ymeans, ss, df)
end
