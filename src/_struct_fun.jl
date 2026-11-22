############---- Data Processing 

# Preprocessing

struct Center{Q<:Float} 
    centers::Vector{Q}
    par::ParCenter
end

struct Scale{Q<:Float} 
    scales::Vector{Q}
    par::ParScale
end

struct Cscale{Q<:Float} 
    centers::Vector{Q}
    scales::Vector{Q}
    par::ParCscale
end

struct Detrendlo
    par::ParDetrendlo
end

struct Detrendpol
    par::ParDetrendpol
end

struct Detrendasls
    par::ParDetrendasls
end

struct Detrendairpls
    par::ParDetrendairpls
end

struct Detrendarpls
    par::ParDetrendarpls
end

struct Emsc{Q<:Float} 
    xref::Vector{Q}
    Xr::Matrix{Q}
    par::ParEmsc
end

struct Fdif
    par::ParFdif
end

struct Mavg
    par::ParMavg
end

struct Savgol
    par::ParSavgol
end

struct Snv
    par::ParSnv
end

struct Snorm
end

struct Rmgap
    par::ParRmgap
end

struct Interpl
    par::ParInterpl
end

# Calibration transfer

struct Calds
    fitm
end

struct Calpds
    fitm
    s::Vector{Vector{Int}}
end

############---- Dimension reduction

struct Pca{Q<:Float}
    T::Matrix{Q} 
    V::Matrix{Q}
    sv::Vector{Q}
    cs::Cscale{Q}
    weights::ProbabilityWeights{Q}
    par::Union{ParPca, ParPcapp, ParPcaout}
end

struct Pcanipals{Q<:Float}
    T::Matrix{Q} 
    V::Matrix{Q}
    sv::Vector{Q}
    cs::Union{Cscale{Q}, NamedTuple}      # pcanipals, pcanipalsmiss
    weights::ProbabilityWeights{Q}
    niter::Union{Nothing, Vector{Int}}    # pcanipals, pcanipalsmiss
    par::ParPcanipals
end

struct Spca{Q<:Float}
    T::Matrix{Q}
    V::Matrix{Q}
    sv::Vector{Q}
    beta::Matrix{Q}
    sel::Vector{Int}
    sellv::Vector{Vector{Int}}
    cs::Cscale{Q}
    weights::ProbabilityWeights{Q}
    niter::Union{Nothing, Vector{Int}}
    par::ParSpca
end


####


struct Kpca{Q<:Float}
    X::Matrix{Q}
    Kt::Adjoint{Q}
    T::Matrix{Q}
    V::Matrix{Q}
    sv::Vector{Q}  
    eig::Vector{Q}    
    DKt::Matrix{Q}
    vtot::Matrix{Q}
    cs::Cscale{Q}
    weights::ProbabilityWeights{Q}
    kwargs::Base.Pairs
    par::ParKpca
end


struct Rp{Q<:Float}
    T::Matrix{Q}
    V::Union{Matrix, SparseArrays.SparseMatrixCSC}
    cs::Cscale{Q}
    par::ParRp
end

struct Umap{Q<:Float} 
    fitm::UMAP.UMAPResult    
    T::Matrix{Q}
    s::Vector{Int}
    cs::Cscale{Q}
    par::ParUmap
end 
    
struct Covsel{Q<:Float}
    sel::Vector{Int}
    selc::Vector{Q}
    C::Matrix{Q}
    xss::Vector{Q}
    yss::Vector{Q}
    xsstot::Q
    ysstot::Q 
    csx::Cscale{Q}
    csy::Cscale{Q}
    weights::ProbabilityWeights{Q}
    par::ParCovsel
end

struct Fda{Q<:Float}
    T::Matrix{Q}
    V::Matrix{Q}
    Tcenters::Matrix{Q}
    eig::Vector{Q}
    sstot::Q
    W::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    cs::Cscale{Q}
    weights::ProbabilityWeights{Q}
    par::ParFda
end

# Multiblock

struct Blockscal{Q<:Float}
    cs::Vector{Cscale{Q}}
    bscales::Vector{Q}
    par::ParBlock
end

struct Mbconcat
    res::Nothing
end

struct Cca{Q<:Float}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Wx::Matrix{Q}
    Wy::Matrix{Q}
    d::Vector{Q}    
    fitm_bl::Blockscal{Q}    
    weights::ProbabilityWeights{Q}
    par::ParCca
end

struct Ccawold{Q<:Float}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Vx::Matrix{Q}
    Vy::Matrix{Q}
    Rx::Matrix{Q}
    Ry::Matrix{Q}    
    Wx::Matrix{Q}
    Wy::Matrix{Q}
    TTx::Vector{Q}
    TTy::Vector{Q}  
    fitm_bl::Blockscal{Q} 
    weights::ProbabilityWeights{Q}
    niter::Vector{Int}
    par::ParCcawold
end

struct Plscan{Q<:Float}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Vx::Matrix{Q}
    Vy::Matrix{Q}
    Rx::Matrix{Q}
    Ry::Matrix{Q}    
    Wx::Matrix{Q}
    Wy::Matrix{Q}
    TTx::Vector{Q}
    TTy::Vector{Q}
    delta::Vector{Q}    
    fitm_bl::Blockscal{Q}   
    weights::ProbabilityWeights{Q}
    par::ParPls2bl
end

struct Plstuck{Q<:Float}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Wx::Matrix{Q}
    Wy::Matrix{Q}
    TTx::Vector{Q}
    TTy::Vector{Q}
    delta::Vector{Q}
    fitm_bl::Blockscal{Q}  
    weights::ProbabilityWeights{Q}
    par::ParPls2bl
end

struct Rasvd{Q<:Float}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Bx::Matrix{Q}
    Wy::Matrix{Q}
    lambda::Vector{Q}    
    fitm_bl::Blockscal{Q}
    weights::ProbabilityWeights{Q}
    par::ParRasvd
end

struct Cpca{Q<:Float}
    T::Matrix{Q}
    U::Matrix{Q}
    W::Matrix{Q}
    Tb::Vector{Matrix{Q}}
    Tbl::Vector{Matrix{Q}}
    Vbl::Vector{Matrix{Q}}
    lb::Matrix{Q}
    mu::Vector{Q}
    fitm_bl::Blockscal{Q}
    weights::ProbabilityWeights{Q}
    niter::Vector{Int}
    par::ParCpca
end

struct Comdim{Q<:Float}
    T::Matrix{Q}
    U::Matrix{Q}
    W::Matrix{Q}
    Tb::Vector{Matrix{Q}}
    Tbl::Vector{Matrix{Q}}
    Vbl::Vector{Matrix{Q}}
    lb::Matrix{Q}
    mu::Vector{Q}
    fitm_bl::Blockscal{Q}
    weights::ProbabilityWeights{Q}
    niter::Vector{Int}
    par::ParCpca
end

############---- Regression

struct Mlr{Q<:Float}
    B::Matrix{Q}   
    int::Matrix{Q}
    weights::ProbabilityWeights{Q}
    par::ParMlr
end

struct Decompx{Q<:Float}
    fit::NamedTuple
    R::Matrix{Q}
    mat::NamedTuple
    ss::NamedTuple
    df::NamedTuple
    f::StatsModels.FormulaTerm
    assign::Vector{Int}
    datf::DataFrame
    xcenters::Vector{Q} 
end

struct Plsr{Q<:Float}
    T::Matrix{Q}
    V::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    csx::Cscale{Q}
    csy::Cscale{Q}
    weights::ProbabilityWeights{Q}
    par::ParPlsr
end

struct Plswold{Q<:Float}
    T::Matrix{Q}
    V::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    csx::Cscale{Q}
    csy::Cscale{Q}
    weights::ProbabilityWeights{Q}
    niter::Union{Nothing, Vector{Int}}   
    par::Union{ParPlswold, ParRrr}
end

struct Cglsr{Q<:Float}
    B::Matrix{Q}
    g::Vector{Q}
    F::Union{Nothing, Matrix{Q}}
    csx::Cscale{Q}
    csy::Cscale{Q}
    par::ParCglsr
end

struct Pcr{Q<:Float}
    fitm::Pca
    C::Matrix{Q}
    csy::Cscale{Q}
    par::ParPcr
end

struct Plsravgunif
    fitm::Plsr
    par::ParPlsravgunif
end

struct Plsravg
    fitm::Plsravgunif
    par::ParPlsravg
end

struct Splsr{Q<:Float}
    T::Matrix{Q}
    V::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    sel::Vector{Int}
    sellv::Vector{Vector{Int}}
    csx::Cscale{Q}
    csy::Cscale{Q}
    weights::ProbabilityWeights{Q}
    niter::Union{Nothing, Vector{Int}}   # snipals_shen when Y with q > 1
    par::ParSplsr
end

struct Spcr{Q<:Float}
    fitm::Spca
    C::Matrix{Q}
    csy::Cscale{Q}
    par::ParSpcr
end

struct Mwplsr{Q<:Float}
    fitm_emb::Vector{Plsr}
    d_emb::Vector{Q}
    w_emb::Vector{Q}
    rescv::DataFrame
    vi::Vector{Q}
    window::Vector{UnitRange{Int}}
    centrw::Vector{Int}
end

struct Kplsr{Q<:Float}
    X::Matrix{Q}
    Kt::Adjoint{Q}
    T::Matrix{Q}
    C::Matrix{Q}
    U::Matrix{Q}
    R::Matrix{Q}
    DKt::Matrix{Q}
    vtot::Matrix{Q}   
    csx::Cscale{Q}
    csy::Cscale{Q}
    weights::ProbabilityWeights{Q}
    iter::Vector{Int}
    kwargs::Base.Pairs
    par::ParKplsr
end

struct Dkplsr{Q<:Float}
    fitm::Plsr
    X::Matrix{Q}
    K::Matrix{Q}
    csx::Cscale{Q}
    csy::Cscale{Q}
    kwargs::Base.Pairs
    par::ParKplsr
end

# Multiblock

struct Mbplsr{Q<:Float}
    fitm::Plsr
    fitm_bl::Blockscal{Q}
    weights::ProbabilityWeights{Q}
    par::ParMbplsr
end

struct Mbplswest{Q<:Float}     
    T::Matrix{Q}
    V::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    Tb::Vector{Matrix{Q}}
    Tbl::Vector{Matrix{Q}}
    Vbl::Vector{Matrix{Q}}
    TT::Vector{Q}
    fitm_bl::Blockscal{Q}
    csy::Cscale{Q}
    weights::ProbabilityWeights{Q}
    lb::Union{Nothing, Matrix{Q}}
    niter::Union{Vector{Int}, Nothing}
    par::ParMbplsr
end

struct Rosaplsr{Q<:Float}
    T::Matrix{Q}
    V::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    bl::Vector{Int}
    fitm_bl::Blockscal{Q}
    csy::Cscale{Q}
    weights::ProbabilityWeights{Q}
    par::ParRosaplsr
end

struct Soplsr{Q<:Float}  
    fitm::Vector{Plsr}
    T::Matrix{Q}
    fit::Matrix{Q}
    b::Vector{Matrix{Q}}
    fitm_bl::Blockscal{Q}  
    csy::Cscale{Q}
    par::ParSoplsr
end

# End

struct Rr{Q<:Float}
    V::Adjoint{Q}
    TtY::Matrix{Q}
    sv::Vector{Q}
    csx::Cscale{Q}
    csy::Cscale{Q}
    weights::ProbabilityWeights{Q}
    par::ParRr
end

struct Rrchol{Q<:Float}
    B::Matrix{Q}   
    int::Matrix{Q}
    weights::ProbabilityWeights{Q}
    par::ParRr
end

struct Krr{Q<:Float}
    X::Matrix{Q}
    K::Matrix{Q}
    U::Matrix{Q}
    UtDY::Matrix{Q}
    sv::Vector{Q}
    DKt::Matrix{Q}
    vtot::Matrix{Q}
    csx::Cscale{Q}
    csy::Cscale{Q}
    weights::ProbabilityWeights{Q}
    kwargs::Base.Pairs
    par::ParKrr
end

# Local

struct Loessr{Q<:Float}
    fitm::Loess.LoessModel
    csx::Cscale{Q}
    par::ParLoessr
end

struct Knnr{Q<:Float}
    X::Matrix{Q}
    Y::Matrix{Q}
    csx::Cscale{Q}
    par::ParKnn
end

struct Lwmlr{Q<:Float}
    X::Matrix{Q}
    Y::Matrix{Q}
    csx::Cscale{Q}
    par::ParLwmlr
end

struct Lwplsr{Q<:Float}
    fitm::Union{Nothing, Plsr}
    X::Matrix{Q}
    Y::Matrix{Q}
    csx::Cscale{Q}
    par::ParLwplsr
end

struct Lwplsravg{Q<:Float}
    fitm::Union{Nothing, Plsr}
    X::Matrix{Q}
    Y::Matrix{Q}
    csx::Cscale{Q}
    par::ParLwplsravg
end

struct Protoplsr{Q<:Float}
    fitm::Vector{Plsr}    
    fitm_emb::Union{Nothing, Plsr}
    Xproto::Matrix{Q}
    Yproto::Matrix{Q}
    coefs::Vector{NamedTuple}
    resnn::NamedTuple
    par::ParProtoplsr
end

struct Protoyclaplsr{Q<:Float}
    fitm::Vector{Plsr}    
    fitm_emb::Union{Nothing, Plsr}
    Xproto::Matrix{Q}
    Yproto::Matrix{Q}
    coefs::Vector{NamedTuple}
    ni::Vector{Int}
    lev::Vector{String}
    par::NamedTuple
end

struct Protoclustplsr
    fitm::Protoyclaplsr
    fitm_emb::Union{Nothing, Plsr}
    fitm_clust::Clustering.KmeansResult
    ycla::Vector{String}
    par::ParProtoclustplsr
end

# Svm, Trees

struct Svmr{Q<:Float64}
    fitm::LIBSVM.SVM
    csx::Cscale{Q}
    par::ParSvm
end

struct Treer{Q<:Float}
    fitm::Union{DecisionTree.Root, DecisionTree.Ensemble}
    featur::Vector{Int}
    csx::Cscale{Q}
    par::Union{ParTree, ParRf}
end

struct Baggr
    fitm
    res_samp::NamedTuple
    q::Int
end

############---- Discrimination

struct Mlrda{Q<:Float}
    fitm_emb::Mlr 
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParMlrda
end

struct Rrda{Q<:Float}
    fitm_emb::Union{Rr, Krr}  
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParRrda, ParKrrda}
end

struct Dmnorm{Q<:Float}
    mu::Vector{Q}
    Uinv::LinearAlgebra.AbstractTriangular{Q} 
    detS::Q
    cst::Q
    par::ParDmnorm
end

struct Dmnormlog{Q<:Float}
    mu::Vector{Q}
    Uinv::LinearAlgebra.AbstractTriangular{Q} 
    logdetS::Q
    logcst::Q
    par::ParDmnorm
end

struct Dmkern{Q<:Float}
    X::Matrix{Q}
    H::Matrix{Q}
    Hinv::Matrix{Q}
    detH::Q
    par::ParDmkern
end

struct Lda{Q<:Float}
    fitm::Vector{Dmnorm}
    W::Matrix{Q}  
    ct::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    weights::ProbabilityWeights{Q}
    par::ParLda
end

struct Qda{Q<:Float}
    fitm::Vector{Dmnorm}
    Wi::Vector{Matrix{Q}}  
    ct::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    weights::ProbabilityWeights{Q}
    par::ParQda
end

struct Kdeda{Q<:Float}
    fitm::Vector{Dmkern}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParKdeda
end

struct Rda{Q<:Float}
    fitm::Vector{Dmnorm}
    Wi::Vector{Matrix{Q}}  
    ct::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    xscales::Vector{Q}
    weights::ProbabilityWeights{Q}
    par::ParRda
end

struct Plsrda{Q<:Float}
    fitm_emb::Union{Plsr, Splsr, Kplsr, Dkplsr} 
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParPlsda, ParSplsda, ParKplsda}
end

struct Plsprobda{Q<:Float}    # plslda, plsqda, plskdeda  
    fitm_emb::Union{Plsr, Splsr, Kplsr, Dkplsr}
    fitm_da::Union{Vector{Lda}, Vector{Qda}, Vector{Kdeda}}  
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParPlsda, ParPlsqda, ParPlskdeda, ParSplsda, ParSplsqda, ParSplskdeda,
        ParKplsda, ParKplsqda, ParKplskdeda}
end

struct Mwplsqda{Q<:Float}
    fitm_emb::Vector{Plsprobda}
    d_emb::Vector{Q}
    w_emb::Vector{Q}
    rescv::DataFrame
    vi::Vector{Q}
    window::Vector{UnitRange{Int}}
    centrw::Vector{Int}
end

struct Mbplsrda{Q<:Float}
    fitm_emb::Mbplsr 
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParMbplsda
end

struct Mbplsprobda{Q<:Float}    # mbplslda, mbplsqda, mbplskdeda  
    fitm_emb::Mbplsr
    fitm_da::Union{Vector{Lda}, Vector{Qda}, Vector{Kdeda}}   
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParMbplsda, ParMbplsqda, ParMbplskdeda}
end

struct Soplsrda{Q<:Float}
    fitm_emb::Soplsr 
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParSoplsda
end

struct Soplsprobda{Q<:Float}    # soplslda, soplsqda, soplskdeda  
    fitm_emb::Soplsr
    fitm_da::Union{Lda, Qda, Kdeda}   
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParSoplsda, ParSoplsqda, ParSoplskdeda}
end

# Local (from below, fitm not yet specified)

struct Knnda{Q<:Float}
    X::Matrix{Q}
    y::Vector{String}
    xscales::Vector{Q}
    ni::Vector{Int}
    lev::Vector{String}
    par::ParKnn
end

struct Lwmlrda{Q<:Float}
    X::Matrix{Q}
    y::Vector{String}
    xscales::Vector{Q}
    ni::Vector{Int}
    lev::Vector{String}
    par::ParLwmlr
end

struct Lwplsrda{Q<:Float}
    fitm::Union{Nothing, Plsr}
    X::Matrix{Q}
    y::Vector{String}
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Union{Nothing, Vector{Q}}
    lev::Vector{String}
    par::ParLwplsda
end

struct Lwplslda{Q<:Float}   
    fitm::Union{Nothing, Plsr}
    X::Matrix{Q}
    y::Vector{String}
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Union{Nothing, Vector{Q}}
    lev::Vector{String}
    par::ParLwplsda
end

struct Lwplsqda{Q<:Float}
    fitm::Union{Nothing, Plsr}
    X::Matrix{Q}
    y::Vector{String}
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Union{Nothing, Vector{Q}}
    lev::Vector{String}
    par::ParLwplsqda
end

# Svm, Trees

struct Svmda{Q<:Float64}
    fitm::LIBSVM.SVM
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParSvm
end

struct Treeda{Q<:Float}
    fitm::Union{DecisionTree.Root, DecisionTree.Ensemble}
    xscales::Vector{Q}
    featur::Vector{Int}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParTree, ParRf}
end

# Occ

struct Occstah{Q<:Float}
    d::DataFrame
    e_cdf::ECDF
    cutoff_d::Q
    res_stah::NamedTuple
    V::Matrix{Q}
    par::ParOccstah
end

struct Occsd{Q<:Float}
    d::DataFrame 
    e_cdf::ECDF
    cutoff_d::Q   
    fitm_emb
    tscales::Vector{Q}
    par::ParOcc
end

struct Occod{Q<:Float}
    d::DataFrame
    e_cdf::ECDF
    cutoff_d::Q   
    fitm_emb
    par::ParOcc
end

struct Occsdod{Q<:Float}
    d::DataFrame
    e_cdf::ECDF
    cutoff_d::Q
    fitm_emb
    fitm_sdod::NamedTuple
    fitm_sd::NamedTuple   
    fitm_od::NamedTuple   
    coefs::Vector{Q}
    par::ParOccsdod
end

struct Occdds{Q<:Float}
    d::DataFrame
    e_cdf::ECDF
    cutoff_d::Q
    nu::Int
    fitm_emb
    res_sd::NamedTuple   
    res_od::NamedTuple   
    coefs::Vector{Q}
    par::ParOccdds
end

struct Occknn{Q<:Float}
    d::DataFrame
    e_cdf::ECDF
    cutoff_d::Q
    X::Matrix{Q}
    xscales::Vector{Q}
    par::ParOccknn
end

struct Occlknn{Q<:Float}
    d::DataFrame
    e_cdf::ECDF
    cutoff_d::Q
    X::Matrix{Q}
    xscales::Vector{Q}
    par::ParOccknn
end

struct Occmwpca{Q<:Float}
    d::Matrix{Q}
    cutoff_d::Vector{Q}
    pwout::Vector{Q}
    cutoff_pwout::Q
    fitm_emb::Vector{Any}
    nlv_emb::Vector{Int}
    fitm_occ::Vector{Occsdod}
    window::Vector{UnitRange{Int}}
    centrw::Vector{Int}
end

struct Occmwpls{Q<:Float}
    d::Matrix{Q}
    cutoff_d::Vector{Q}
    pwout::Vector{Q}
    cutoff_pwout::Q
    fitm_emb::Vector{Any}
    nlv_emb::Vector{Int}
    fitm_occ::Vector{Occsdod}
    window::Vector{UnitRange{Int}}
    centrw::Vector{Int}
end
