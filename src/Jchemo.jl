module Jchemo  # Start-Module

using Clustering 
using DataInterpolations  # 1D interpolations (interpl) 
using DecisionTree
using Distributions
using DataFrames
using Distances
using ImageFiltering      # convolutions in preprocessing (mavg, savgol), alternative = DSP.jl
using LIBSVM 
using LinearAlgebra
using Loess
using Makie
using NearestNeighbors
using Random
using SparseArrays 
using Statistics
using StatsBase           # countmap, ecdf, sample etc.
using StatsAPI
using StatsModels
using UMAP

include("_const.jl")

# The order below is required
include("_struct_param.jl")
include("_struct_fun.jl")      
include("_model_work.jl")
include("_pip.jl")
# End

include("_defaults.jl")

######---- Misc

include("_util.jl")
include("_util_colwise.jl")
include("_util_mb.jl")
include("_util_recod.jl")
include("_util_rowwise.jl")
include("_util_center_scale.jl")
include("_util_stat.jl")
include("_util_table.jl")
include("_util_weighting.jl")

include("repc.jl")

##
include("angles.jl")
include("colmedspa.jl")
include("ellipse.jl")
include("matW.jl")
include("nipals.jl")
include("nipalsmiss.jl")
include("simpp.jl")
include("snipals_post.jl")
include("snipals_shen.jl")
include("boxcox.jl")

######---- Preprocessing

include("center_scale.jl") 
include("preprocessing.jl") 
include("detrend_asls.jl") 
include("detrend_airpls.jl") 
include("detrend_arpls.jl") 
include("rmgap.jl")

######---- Graphics

include("plotsp.jl")
include("plotxy.jl")
include("plotxyz.jl")
include("plotlv.jl")
include("plotgrid.jl")
include("plotconf.jl")

######---- Dimension reduction and exploratory

include("fda.jl")      # here since ::Fda called in pcasvd
include("fdasvd.jl")     
include("pcasvd.jl")
include("pcaeigen.jl")
include("pcaeigenk.jl")
include("pcanipals.jl")
include("pcanipalsmiss.jl")
include("spca.jl")
include("pcasph.jl") 
include("pcapp.jl") 
include("pcaout.jl") 
include("kpca.jl")

include("xfit.jl")
include("xresid.jl")

include("rpmat.jl")
include("rp.jl")
include("umap.jl")

include("cca.jl")
include("ccawold.jl")
include("plscan.jl")
include("plstuck.jl")
include("rasvd.jl")
include("cpca.jl")
include("comdim.jl")

include("covsel.jl")

######---- Regression 

include("mlr.jl")

include("aov1.jl")
include("manova.jl")
include("decompx.jl")
include("asca.jl")
include("emm.jl")
include("hotelling.jl")
include("wilks.jl")
include("waldtest.jl")

include("plskern.jl")
include("plsnipals.jl")
include("plswold.jl") 
include("plsrosa.jl")
include("plssimp.jl")
include("plsravg.jl")
include("plsravg_unif.jl")

include("cglsr.jl")
include("pcr.jl")
include("rrr.jl")

include("splsr.jl")
include("spcr.jl")

include("mwplsr.jl")

include("plsrout.jl")

include("kplsr.jl")
include("dkplsr.jl")

include("mbplsr.jl") 
include("mbplswest.jl")
include("rosaplsr.jl") 
include("soplsr.jl")

include("vip.jl") 

include("rr.jl")
include("rrchol.jl")
include("krr.jl")

include("loessr.jl")
include("locw.jl")
include("locwlv.jl")
include("knnr.jl")
include("lwmlr.jl")
include("lwplsr.jl")
include("lwplsravg.jl")

include("protoplsr.jl")
include("protoyclaplsr.jl")
include("protoclustplsr.jl")

include("svmr.jl")
include("treer.jl")
include("rfr.jl")

include("sampbag.jl")
include("baggr.jl")

######---- Discrimination 

include("mlrda.jl")
include("plsrda.jl") 
include("rrda.jl")
include("splsrda.jl")
include("kplsrda.jl")
include("dkplsrda.jl")
include("krrda.jl")
include("mbplsrda.jl") 

include("lwmlrda.jl")
include("lwplsrda.jl")

include("lda.jl")
include("qda.jl")
include("rda.jl")
include("kdeda.jl")

include("plslda.jl")
include("plsqda.jl")
include("plskdeda.jl")

include("splslda.jl")
include("splsqda.jl")
include("splskdeda.jl")

include("kplslda.jl")
include("kplsqda.jl")
include("kplskdeda.jl")

include("dkplslda.jl")
include("dkplsqda.jl")
include("dkplskdeda.jl")

include("mbplslda.jl") 
include("mbplsqda.jl") 
include("mbplskdeda.jl") 

include("lwplslda.jl")
include("lwplsqda.jl")

include("knnda.jl")

include("svmda.jl")
include("treeda.jl")
include("rfda.jl")

######---- Occ

include("outstah.jl")
include("outeucl.jl")
include("pcout.jl")
include("outsd.jl")
include("outod.jl")
include("outsdod.jl")
include("outknn.jl")
include("outlknn.jl")

include("occsd.jl")
include("occod.jl") 
include("occsdod.jl")
include("occdds.jl")
include("occstah.jl")
include("occknn.jl")
include("occlknn.jl")
include("occmwpca.jl")
include("occmwpls.jl")

######---- Variable importance

include("viperm.jl")

######---- Tuning models

include("mpar.jl")

include("conf.jl")
include("segmkf.jl")
include("segmts.jl")

include("gridscore.jl")
include("gridscore_br.jl")
include("gridscore_lv.jl")
include("gridscore_lb.jl")

include("gridcv.jl")

include("predictcv.jl")

include("scores_reg.jl")
include("scores_da.jl")

include("dfplsr_cg.jl")
include("aicplsr.jl")
include("selwold.jl")
include("durbinw.jl")

######---- Distributions

include("dmnorm.jl")
include("dmnormlog.jl")
include("dmkern.jl")

######---- Calibration transfer

include("calds.jl")
include("calpds.jl")
include("difmean.jl")
include("eposvd.jl")

######---- Sampling

include("sampks.jl")
include("sampdp.jl")
include("sampwsp.jl")
include("samprand.jl")
include("sampsys.jl")
include("sampcla.jl")
include("sampdatf.jl")

include("distances.jl")
include("getknn.jl")
include("wdis.jl") 
include("wtal.jl") 
include("winvs.jl")
include("kernels.jl")

export 
    model,
    modelx, modelxy, 
    fit!,
    transf!,
    pip,
    
    ######---- Utilities
    @head, @pmod, @names, @pars, @plist, @type,
    #
    aggmean, aggstat, 
    aggsumv,  
    sumv, meanv, medv, stdv, varv, madv, iqrv, normv, norm2v, kurtv, 
    entrv, quantv,
    boxcox, boxcox_transf, boxcox_transf!,
    colsum, colmean, colmed, colvar, colstd, colprt, colmad, colnorm2, colnorm, colkurt, 
    colentr, colquant,
    colsumskip, colmeanskip, colstdskip, colvarskip,
    rowsum, rowmean, rowvar, rowstd, rownorm2, rownorm,
    rowsumskip, rowmeanskip, rowstdskip, rowvarskip,
    def_colscal,
    convertdf,
    covv, covm, 
    corv, corm,
    cosv, cosm, 
    dummy,
    dupl, findmiss,
    ensure_df, ensure_mat, ensure_mat_mb, 
    fblockscal, fblockscal!,
    fcenter, fcenter!, 
    fcscale, fcscale!, 
    fweightr, fweightr!,
    fweightc, fweightc!,
    findmax_cla, 
    frob, frob2, 
    fscale, fscale!,
    wdis, wtal, 
    list, 
    matB, matW, matWc, 
    mblock,
    mlev,
    pweight, pweightcla,
    nipals,
    nipalsmiss,
    nro, nco, 
    out,
    parsemiss,
    pval,
    recovkw,
    recod_catbydict,
    recod_catbyind, 
    recod_catbyind2, 
    recod_catbylev, 
    recod_contbylev,     
    recod_indbylev, 
    recod_miss, 
    expand_tab2d, expand_grid,
    rmcol, rmrow, 
    finduniq,
    simpphub, simppsph,   
    snipals_shen,
    thresh_soft, thresh_hard, 
    softmax,
    sourcedir,
    summ,
    mbin,
    tab, tabcont, tabdupl,
    vcatdf,
    vcol, vrow,

    ######---- Distributions
    dmnorm, dmnorm!,
    dmnormlog, dmnormlog!,
    dmkern,
    
    ######---- Pre-processing
    detrend_pol, detrend_lo,  
    detrend_asls, detrend_airpls, detrend_arpls,
    fdif,
    interpl, 
    center, scale, cscale,
    repc, repc!,
    blockscal,
    #cubic_spline,
    mavg, 
    msc, emsc,
    rmgap,
    savgk, savgol,
    snorm,
    snv, 
    
    ######---- Calibration ransfer
    calds, calpds,
    difmean,
    eposvd,
    
    ######---- Dimension reduction and exploratory
    pcasvd, pcasvd!, 
    pcaeigen, pcaeigen!, 
    pcaeigenk, pcaeigenk!,
    pcanipals, pcanipals!,
    pcanipalsmiss, pcanipalsmiss!,
    pcasph, pcasph!,
    pcapp, pcapp!,
    pcaout, pcaout!,
    spca, spca!,
    kpca,
    rpmatgauss, rpmatli, rp, rp!,
    umap,
    # Multiblock
    rd, rv, 
    mbconcat, fconcat,
    cca, cca!,
    ccawold, ccawold!,
    plscan, plscan!,
    plstuck, plstuck!,
    rasvd, rasvd!,
    cpca, cpca!,
    comdim, comdim!,
    # Fcatorial DA
    fda, fda!, 
    fdasvd, fdasvd!,
    # Partial covariances
    covsel,
    
    ######---- Regression
    # Mlr
    mlr, mlr!, 
    # Anova related
    aov1, 
    manova, decompx, asca, 
    emm, 
    permut, 
    hotelling, waldtest, wilks,
    # Plsr
    plskern, plskern!, 
    plsnipals, plsnipals!, 
    plsrosa, plsrosa!, 
    plssimp, plssimp!,
    plswold, plswold!,
    plsravg, plsravg!,
    cglsr, cglsr!,
    rrr, rrr!,   
    splsr, splsr!, 
    mwplsr,
    plsrout, plsrout!,
    kplsr, kplsr!,
    dkplsr, dkplsr!,
    dfplsr_cg, aicplsr,     
    mbplsr, mbplsr!,
    mbplswest, mbplswest!,
    rosaplsr, rosaplsr!,
    soplsr,
    vip,
    # Ridge
    rr, rr!, rrchol, rrchol!,
    # Pcr
    pcr,
    spcr, spcr!,
    krr, krr!, 
    # Local
    loessr,
    locw, locwlv,
    knnr,
    lwmlr,
    lwplsr, lwplsravg,
    protoplsr, protoclustplsr,
    # Svm and trees
    svmr,
    treer, rfr, 
    # Bagging
    sampbag, 
    baggr, 
    vi_baggr,
    # Utils
    transf, coef, coefmatb, predict,
    transfbl, 
    xfit, xfit!, xresid, xresid!,

    ######---- Discrimination
    mlrda,
    rrda, krrda,
    lda, qda, kdeda,
    rda,
    plsrda,
    plslda, plsqda, plskdeda,
    kplsrda, 
    kplslda, kplsqda, kplskdeda, 
    dkplsrda,
    dkplslda, dkplsqda, dkplskdeda, 
    svmda, 
    treeda, rfda,
    # Sparse 
    splsrda,
    splslda, splsqda, splskdeda,
    # Multiblock
    mbplsrda, 
    mbplslda, mbplsqda, mbplskdeda,
    # Local 
    lwmlrda,
    lwplsrda, 
    lwplslda, lwplsqda,
    knnda,
    # One-class
    outstah, outeucl, 
    pcout,
    outsd, outod, outsdod,
    outknn, outlknn,
    occsd, occod, occsdod, 
    occdds,
    occstah,
    occknn, occlknn,
    occmwpca,
    occmwpls,
    
    ######---- Validation
    residreg, residcla, 
    ssr, msep, rmsep, rmsepstand, rmseprel, mae,
    bias, sep, cor2, r2, rpd, rpdr, mse, 
    errp, merrp,
    mpar,
    segmts, segmkf,
    gridscore, 
    gridcv, 
    predictcv,
    selwold, 
    durbinw, 
    conf, 

    ######---- Variable importance 
    viperm!,

    ######---- Sampling data
    sampks, sampdp, sampwsp, samprand, sampsys, sampcla, 
    sampdatf,

    ######---- Distances
    getknn, 
    wdis, wtal, winvs, winvs!,
    eucl2, mah2, mah2chol,
    krbf, kpol,

    ######---- Graphics
    plotsp,
    plotxy, plotxyz,
    plotlv,
    plotgrid, 
    plotconf
    # Not exported since surcharge:
    # - summary => Base.summary

end # End-Module




