# DTLC
The code is for paper: Discriminative Transfer Feature and Label Consistency for Cross-Domain Image Classification

1. The data folder has all the datasets used in this paper: CMU-PIE, ImageNet+VOC2007, Office+Caltech10_DeCAF6, Office+Caltech10_SURF and Office-31_DeCAF7.

2. The folder code_TABLE_II_III_IV_V_VI_VII has all the codes for Table II~Table VII in our paper. To be specific, you can obtain the results of DTLC on dataset CMU-PIE, Office31_DeCAF7, Office+Caltech10_DeCAF6, Office+Caltech10_SURF and Imagenet+VOC2007 by running table2_run_CMUPIE.m, table3_table7_run_Office31_DeCAF7.m, table4_table7_run_OfficeCaltech10_DeCAF6.m table5_run_OfficeCaltech10_SURF.m and table6_run_ImagenetVOC2007.m, respectively.

3. The folder Fig4_tsne is for t-SNE visualization of source and target data for task C07-C29 of CMU-PIE. For example, in the DTLC_tsne_save_data folder, we can run run_pie.m to save projected data in the folder of save_data. Then run tsne_visualization.m to get the t-SNE visualization of DTLC. This way is fitting for all the other methods. We can also run map_visual.m to show all the t-SNE visualizations for all the methods in Fig4 of the paper.

4. The folder Fig5_weight is the code for Fig. 5 in our paper. Run Fig5_run_decaf6.m in the folder DTLC_fig5_weight_code to produce the accuracy of every 1/3 target data with different weight value from small to large. 

5. The folder Fig6_JDA_DT_DTLC is for Fig.6 in our paper: results of JDA, DTLC w/o label consistency and DTLC. The folders JDAcode, DTLC_wo_LC and DTLCcode are for methods JDA, DTLC w/o label consistency and DTLC, respectively.

6. The folder Fig7_DTLC_iteration is for Fig. 7 to illustrate the iterative optimization process in DTLC for 10 iterations. The folder DICDcode is the code of method DICD.

7. The folder Fig8_DTLC_parameter_sensitivity is for Fig. 8: parameter sensitivity studies w.r.t. \alpha, \beta and \eta, respectively. We can run run_DATASET_alpha.m, run_ DATASET _beta.m, and run_ DATASET_eta.m to test the parameter sensitivity of \alpha, \beta and \eta to DTLC.

8. The folder TABLEVIII is for Table VIII in our paper. Folder DTLC_wo_LC_code is the code of DTLC without LC. Run run_Office31_DeCAF7.m to get the results.

9. The folder TABLEIX_DTLC_Variants is for Table IX in our paper to compare the results of DTLC and its variants. Folder DTLCcode_random is the code of DTLC-random, and folder JDAcode_LC is the code of JDA-LC. We can run run_DATASET.m to get the results of each method.



