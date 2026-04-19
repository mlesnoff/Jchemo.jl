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

## The order below is required
include("_struct_param.jl")
include("_struct_fun.jl")      
include("_model_work.jl")
include("_pip.jl")
## End
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

######---- Distributions

include("dmnorm.jl")
include("dmnormlog.jl")
include("dmkern.jl")

######---- Exploratory

include("fda.jl")  # Here since ::Fda called in pcasvd
include("fdasvd.jl")     
include("pcasvd.jl")
include("pcaeigen.jl")
include("pcaeigenk.jl")
include("pcanipals.jl")
include("pcanipalsmiss.jl")
include("pcasph.jl") 
include("pcapp.jl") 
include("pcaout.jl") 
include("kpca.jl")
include("covsel.jl")
include("rpmat.jl")
include("rp.jl")
include("umap.jl")

## Sparse
include("spca.jl")

## Multiblock 
include("cca.jl")
include("ccawold.jl")
include("plscan.jl")
include("plstuck.jl")
include("rasvd.jl")
include("mbpca.jl")
include("comdim.jl")

######---- Regression 

include("mlr.jl")
include("rr.jl")
include("rrchol.jl")
include("pcr.jl")
include("rrr.jl") 
include("plskern.jl")
include("plsnipals.jl")
include("plswold.jl") 
include("plsrosa.jl")
include("plssimp.jl")
include("cglsr.jl")
include("plsrout.jl")
include("plsravg.jl")
include("plsravg_unif.jl")
include("krr.jl")
include("kplsr.jl")
include("dkplsr.jl")

include("aov1.jl")
include("manova.jl")
include("decompx.jl")
include("asca.jl")
include("emm.jl")
include("hotelling.jl")
include("wilks.jl")
include("waldtest.jl")

include("dfplsr_cg.jl")
include("aicplsr.jl")
include("vip.jl") 

include("xfit.jl")
include("xresid.jl")

## Sparse
include("splsr.jl")
include("spcr.jl")

## Multiblock
include("mbplsr.jl") 
include("mbplswest.jl")
include("rosaplsr.jl") 
include("soplsr.jl") 

## Local
include("locw.jl")
include("locwlv.jl")
include("knnr.jl")
include("lwmlr.jl")
include("lwplsr.jl")
include("lwplsravg.jl")
include("loessr.jl")

## Validation
include("mpar.jl")
include("scores.jl")
include("conf.jl")
include("segmkf.jl")
include("segmts.jl")

include("gridscore.jl")
include("gridscore_br.jl")
include("gridscore_lv.jl")
include("gridscore_lb.jl")

include("gridcv.jl")

include("predictcv.jl")

include("selwold.jl")

## Variable importance (direct methods) 
include("isel.jl")
include("viperm.jl")

## Svm, Trees
include("svmr.jl")
include("treer.jl")
include("rfr.jl")

## Bagging 

include("sampbag.jl")
include("baggr.jl")

## Prototypes

include("protoplsr.jl")
include("protoyclaplsr.jl")
include("protoclustplsr.jl")
include("rclustplsr.jl")

######---- Discrimination 

include("lda.jl")
include("qda.jl")
include("rda.jl")
include("kdeda.jl")
include("mlrda.jl")
include("rrda.jl")
include("plsrda.jl") 
include("plslda.jl")
include("plsqda.jl")
include("plskdeda.jl")
include("krrda.jl")
include("kplsrda.jl")
include("kplslda.jl")
include("kplsqda.jl")
include("kplskdeda.jl")
include("dkplsrda.jl")
include("dkplslda.jl")
include("dkplsqda.jl")
include("dkplskdeda.jl")

## Sparse
include("splsrda.jl")
include("splslda.jl")
include("splsqda.jl")
include("splskdeda.jl")

## Multiblock
include("mbplsrda.jl") 
include("mbplslda.jl") 
include("mbplsqda.jl") 
include("mbplskdeda.jl") 

## One-class
include("outsd.jl")
include("outod.jl")
include("outsdod.jl")
include("outstah.jl")
include("outeucl.jl")
include("outknn.jl")
include("outlknn.jl")

include("occsd.jl")
include("occod.jl") 
include("occsdod.jl")
include("occdds.jl")
include("occstah.jl")
include("occknn.jl")
include("occlknn.jl")

## Local
include("lwmlrda.jl")
include("lwplsrda.jl")
include("lwplslda.jl")
include("lwplsqda.jl")
include("knnda.jl")

## Svm, Trees
include("svmda.jl")
include("treeda.jl")
include("rfda.jl")

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
include("sampdf.jl")

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
    ##
    aggmean, aggstat, 
    aggsumv,  
    sumv, meanv, stdv, varv, medv, madv, iqrv, normv, norm2v,
    colsum, colmean, colnorm, colnorm2, colstd, colvar, colmed, colmad, 
    colsumskip, colmeanskip, colstdskip, colvarskip,
    convertdf,
    covv, covm, 
    corv, corm,
    cosv, cosm, 
    dummy,
    dupl, findmiss,
    ensure_df, ensure_mat,
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
    recod_catbyint, 
    recod_catbylev, 
    recod_contbyint,     
    recod_indbylev, 
    recod_miss, 
    expand_tab2d, expand_grid,
    rmcol, rmrow, 
    finduniq,
    rowsum, rowmean, rownorm, rownorm2, rowstd, rowvar,
    rowsumskip, rowmeanskip, rowstdskip, rowvarskip,
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
    ## Pre-processing
    detrend_pol, detrend_lo,  
    detrend_asls, detrend_airpls, detrend_arpls,
    fdif,
    interpl, 
    center, scale, cscale,
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
    ######---- Exploratory
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
    covsel,
    rpmatgauss, rpmatli, rp, rp!,
    umap,
    ## Multiblock
    rd, rv, 
    mbconcat, fconcat,
    cca, cca!,
    ccawold, ccawold!,
    plscan, plscan!,
    plstuck, plstuck!,
    rasvd, rasvd!,
    mbpca, mbpca!,
    comdim, comdim!, 
    ######---- Regression
    mlr, mlr!, mlrchol, mlrchol!, 
    mlrpinv, mlrpinv!, mlrpinvn, mlrpinvn!,
    mlrvec, mlrvec!,
    rr, rr!, rrchol, rrchol!,
    pcr,
    plskern, plskern!, 
    plsnipals, plsnipals!, 
    plsrosa, plsrosa!, 
    plssimp, plssimp!,
    plswold, plswold!,
    cglsr, cglsr!,
    plsrout, plsrout!,
    rrr, rrr!,   
    krr, krr!, kplsr, kplsr!, 
    dkplsr, dkplsr!,
    plsravg, plsravg!,
    dfplsr_cg, aicplsr,
    svmr,
    treer, rfr, 
    ## Anova
    aov1, 
    manova, decompx, asca, 
    emm, 
    permut, 
    hotelling, waldtest, wilks,
    ## Sparse 
    spcr, spcr!,
    splsr, splsr!, 
    ## Multi-block
    mbplsr, mbplsr!,
    mbplswest, mbplswest!,
    rosaplsr, rosaplsr!,
    soplsr,
    ## Local
    locw, locwlv,
    knnr,
    lwmlr,
    lwplsr, lwplsravg,
    loessr,
    ## Bagging
    baggr,
    ## Prototype
    protoplsr, protoclustplsr, rclustplsr,
    ## Variable selection/importance (direct methods) 
    vip, 
    viperm!,
    isel!,
    ## Utils
    xfit, xfit!, xresid, xresid!,
    ######---- Discrimination
    fda, fda!, fdasvd, fdasvd!,
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
    ## One-class
    outstah, outeucl,
    outsd, outod, outsdod,
    outknn, outlknn,
    occsd, occod, occsdod, 
    occdds,
    occstah,
    occknn, occlknn,
    ## Sparse 
    splsrda,
    splslda, splsqda, splskdeda,
    ## Local 
    lwmlrda,
    lwplsrda, 
    lwplslda, lwplsqda,
    knnda,
    ## Multiblock
    mbplsrda, 
    mbplslda, mbplsqda, mbplskdeda,
    ## Auxiliary
    transf, coef, predict,
    transfbl, 
    ## Validation
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
    conf, 
    ######---- Sampling
    sampks, sampdp, sampwsp, samprand, sampsys, sampcla, 
    sampdf,
    sampbag, 
    ######---- Distances
    getknn, wdis, wtal, winvs, winvs!,
    eucl2, mah2, mah2chol,
    krbf, kpol,
    ######---- Graphics
    plotsp,
    plotxy, plotxyz,
    plotlv,
    plotgrid, 
    plotconf
    ## Not exported since surcharge:
    ## - summary => Base.summary

end # End-Module




