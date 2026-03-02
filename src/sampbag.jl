function sampbag(X; rep = 50, rowsamp = .7, replace = true, colsamp = 1)
    X = ensure_mat(X)
    n, p = size(X) 
    range_n = collect(1:n)
    range_p = collect(1:p) 
    ##
    mrow = Int(round(rowsamp * n))
    moob = n - mrow
    mcol = max(1, Int(round(colsamp * p)))    
    ##
    srow = list(Vector{Int}, rep)
    srow_oob = list(Vector{Int}, rep)
    scol = list(Vector{Int}, rep)
    ##
    ordered = true
    Threads.@threads for i = 1:rep 
        ## Rows
        s = StatsBase.sample(range_n, mrow; replace, ordered)
        srow[i] = s
        srow_oob[i] = range_n[setdiff(1:end, s)]
        ## Columns
        if colsamp == 1
            scol[i] = range_p
        else
            s = StatsBase.sample(range_p, mcol; replace = false, ordered)
            scol[i] = s
        end
    end
    (srow = srow, srow_oob, scol)
end

function sampbag(X, colweight::ProbabilityWeights; rep = 50, rowsamp = .7, replace = true, colsamp = 1)
    X = ensure_mat(X)
    n, p = size(X) 
    range_n = collect(1:n)
    range_p = collect(1:p) 
    ##
    mrow = Int(round(rowsamp * n))
    moob = n - mrow
    mcol = max(1, Int(round(colsamp * p)))    
    ##
    srow = list(Vector{Int}, rep)
    srow_oob = list(Vector{Int}, rep)
    scol = list(Vector{Int}, rep)
    ##
    ordered = true
    Threads.@threads for i = 1:rep 
        ## Rows
        s = StatsBase.sample(range_n, mrow; replace, ordered)
        srow[i] = s
        srow_oob[i] = range_n[setdiff(1:end, s)]
        ## Columns
        if colsamp == 1
            scol[i] = range_p
        else
            colweight.values[colweight.values .== 0] .= eps(eltype(colweight.values))
            colweight = pweight(colweight.values)
            s = StatsBase.sample(range_p, colweight, mcol; replace = false, ordered)
            scol[i] = s
        end
    end
    (srow = srow, srow_oob, scol)
end


