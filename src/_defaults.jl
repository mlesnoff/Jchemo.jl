function defaults(algo::Function)

    defs = [

        detrend_pol Jchemo.ParDetrendPol ;
        detrend_lo Jchemo.ParDetrendLo ;
        detrend_asls Jchemo.ParDetrendAsls ;
        detrend_airpls Jchemo.ParDetrendAirpls ;
        detrend_arpls Jchemo.ParDetrendArpls ;
        fdif Jchemo.ParFdif ;
        interpl Jchemo.ParInterpl ;
        mavg Jchemo.ParMavg ;
        savgol Jchemo.ParSavgol ;
        snv Jchemo.ParSnv ;
        rmgap Jchemo.ParRmgap ;

        nipals Jchemo.ParNipals ;
        nipalsmiss Jchemo.ParNipals ;
        snipals_shen Jchemo.ParSnipals ;

        pcasvd Jchemo.ParPca ; 
        pcaeigen Jchemo.ParPca ; 
        pcaeigenk Jchemo.ParPca ; 
        pcasph Jchemo.ParPca ;
        pcanipals Jchemo.ParPcanipals ; 
        pcanipalsmiss Jchemo.ParPcanipals ;
        pcapp Jchemo.ParPcapp ;
        pcaout Jchemo.ParPcaout ;

        covsel Jchemo.ParCovsel ;
        
        rp Jchemo.ParRp ;

        spca Jchemo.ParSpca ;

        kpca Jchemo.ParKpca ;
        umap Jchemo.ParUmap ;

        fda Jchemo.ParFda ; 
        fdasvd Jchemo.ParFda ;

        blockscal Jchemo.ParBlock ;
        plscan Jchemo.ParPls2bl ; 
        plstuck Jchemo.ParPls2bl ;
        rasvd Jchemo.ParRasvd ;
        rrr Jchemo.ParRrr ;

        mbpca Jchemo.ParMbpca ; 
        comdim Jchemo.ParMbpca ;
        cca Jchemo.ParCca ;
        ccawold Jchemo.ParCcawold ;

        ##

        mlr Jchemo.ParMlr ; 
        mlrpinv Jchemo.ParMlr ; 
        mlrvec Jchemo.ParMlr ;

        rr Jchemo.ParRr ; 
        rrchol Jchemo.ParRr ;

        pcr Jchemo.ParPcr ;

        plskern Jchemo.ParPlsr ; 
        plsnipals Jchemo.ParPlsr ; 
        plswold Jchemo.ParPlsr ; 
        plsrosa Jchemo.ParPlsr ; 
        plssimp Jchemo.ParPlsr ; 
        plsravg Jchemo.ParPlsr ; 
        plsravg_unif Jchemo.ParPlsr ;
        cglsr Jchemo.ParCglsr ; 
        aicplsr Jchemo.ParCglsr ;
        plsrout Jchemo.ParPlsrout ;

        spcr Jchemo.ParSpca ;
        splsr Jchemo.ParSplsr ;

        krr Jchemo.ParKrr ;
        kplsr Jchemo.ParKplsr ; 
        dkplsr Jchemo.ParKplsr ;

        svmr Jchemo.ParSvm ; 
        svmda Jchemo.ParSvm ;

        treer Jchemo.ParTree ; 
        treeda Jchemo.ParTree ;
        rfr Jchemo.ParRf ; 
        rfda Jchemo.ParRf ;

        knnr Jchemo.ParKnn ;
        lwmlr Jchemo.ParLwmlr ;
        lwplsr Jchemo.ParLwplsr ; 
        lwplsravg Jchemo.ParLwplsr ;
        loessr Jchemo.ParLoessr ;

        protoplsr Jchemo.Parprotoplsr ;
        protoclustplsr Jchemo.Parprotoclustplsr ;
        rclustplsr Jchemo.Parrclustplsr ;

        mbplsr Jchemo.ParMbplsr ; 
        mbplswest Jchemo.ParMbplsr ;
        rosaplsr Jchemo.ParSoplsr ; 
        soplsr Jchemo.ParSoplsr ;

        ##

        dmnorm Jchemo.ParDmnorm ; 
        dmnormlog Jchemo.ParDmnorm ;
        dmkern Jchemo.ParDmkern ;

        lda Jchemo.ParLda ;
        qda Jchemo.ParQda ;
        rda Jchemo.ParRda ;
        kdeda Jchemo.ParKdeda ;
        
        mlrda Jchemo.ParMlrda ;
        rrda Jchemo.ParRrda ;
        plsrda Jchemo.ParPlsda ; 
        plslda Jchemo.ParPlsda ;
        plsqda Jchemo.ParPlsqda ;
        plskdeda Jchemo.ParPlskdeda ;

        splsrda Jchemo.ParSplsda ; 
        splslda Jchemo.ParSplsda ;
        splsqda Jchemo.ParSplsqda ;
        splskdeda Jchemo.ParSplskdeda ;

        krrda Jchemo.ParKrrda ;
        kplsrda Jchemo.ParKplsda ; 
        kplslda Jchemo.ParKplsda ; 
        dkplsrda Jchemo.ParKplsda ; 
        dkplslda Jchemo.ParKplsda ;
        kplsqda Jchemo.ParKplsqda ;
        dkplsqda Jchemo.ParKplsqda ;
        kplskdeda Jchemo.ParKplskdeda ; 
        dkplskdeda Jchemo.ParKplskdeda ;

        knnda Jchemo.ParKnn ;
        lwmlrda Jchemo.ParLwmlr ;
        lwplsrda Jchemo.ParLwplsda ; 
        lwplslda Jchemo.ParLwplsda ;
        lwplsqda Jchemo.ParLwplsqda ;

        mbplsrda Jchemo.ParMbplsda ; 
        mbplslda Jchemo.ParMbplsda ;
        mbplsqda Jchemo.ParMbplsqda ;
        mbplskdeda Jchemo.ParMbplskdeda ;

        ##

        occsd Jchemo.ParOcc ; 
        occod Jchemo.ParOcc ;
        occsdod Jchemo.ParOccsdod ;
        occdds Jchemo.ParOccdds ;
        occstah Jchemo.ParOccstah ;
        occknn Jchemo.ParOccknn ; 
        occlknn Jchemo.ParOccknn 

        ]

    s = in((algo,)).(defs[:, 1]) 
    first(defs[s, 2])

end

