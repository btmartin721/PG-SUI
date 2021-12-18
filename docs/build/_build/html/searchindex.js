Search.setIndex({docnames:["index","install","pgsui","pgsui.impute","pgsui.impute.estimators","pgsui.impute.impute","pgsui.impute.iterative_imputer_fixedparams","pgsui.impute.iterative_imputer_gridsearch","pgsui.impute.neural_network_imputers","pgsui.impute.simple_imputers","pgsui.read_input","pgsui.read_input.popmap_file","pgsui.read_input.read_input","pgsui.utils","pgsui.utils.misc","pgsui.utils.sequence_tools","tutorial"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["index.rst","install.rst","pgsui.rst","pgsui.impute.rst","pgsui.impute.estimators.rst","pgsui.impute.impute.rst","pgsui.impute.iterative_imputer_fixedparams.rst","pgsui.impute.iterative_imputer_gridsearch.rst","pgsui.impute.neural_network_imputers.rst","pgsui.impute.simple_imputers.rst","pgsui.read_input.rst","pgsui.read_input.popmap_file.rst","pgsui.read_input.read_input.rst","pgsui.utils.rst","pgsui.utils.misc.rst","pgsui.utils.sequence_tools.rst","tutorial.rst"],objects:{"":[[2,0,0,"-","pgsui"]],"pgsui.impute":[[4,0,0,"-","estimators"],[5,0,0,"-","impute"],[6,0,0,"-","iterative_imputer_fixedparams"],[7,0,0,"-","iterative_imputer_gridsearch"],[8,0,0,"-","neural_network_imputers"],[9,0,0,"-","simple_imputers"]],"pgsui.impute.estimators":[[4,1,1,"","ImputeBayesianRidge"],[4,1,1,"","ImputeGradientBoosting"],[4,1,1,"","ImputeKNN"],[4,1,1,"","ImputeLightGBM"],[4,1,1,"","ImputeMF"],[4,1,1,"","ImputeNLPCA"],[4,1,1,"","ImputeRandomForest"],[4,1,1,"","ImputeUBP"],[4,1,1,"","ImputeVAE"],[4,1,1,"","ImputeXGBoost"]],"pgsui.impute.estimators.ImputeMF":[[4,2,1,"","fit_predict"]],"pgsui.impute.estimators.ImputeNLPCA":[[4,3,1,"","best_params"],[4,3,1,"","imputed"],[4,4,1,"","nlpca"]],"pgsui.impute.estimators.ImputeUBP":[[4,3,1,"","best_params"],[4,3,1,"","imputed"],[4,4,1,"","nlpca"]],"pgsui.impute.estimators.ImputeVAE":[[4,3,1,"","best_params"],[4,3,1,"","imputed"]],"pgsui.impute.impute":[[5,1,1,"","Impute"]],"pgsui.impute.impute.Impute":[[5,2,1,"","_average_list_of_dicts"],[5,2,1,"","_defile_dataset"],[5,2,1,"","_define_iterative_imputer"],[5,2,1,"","_format_features"],[5,2,1,"","_gather_impute_settings"],[5,2,1,"","_get_best_params"],[5,2,1,"","_impute_df"],[5,2,1,"","_impute_eval"],[5,2,1,"","_impute_gridsearch"],[5,2,1,"","_impute_single"],[5,2,1,"","_imputer_validation"],[5,2,1,"","_mode_list_of_dicts"],[5,2,1,"","_print_scores"],[5,2,1,"","_remove_invalid_cols"],[5,2,1,"","_remove_nonbiallelic"],[5,2,1,"","_subset_data_for_gridsearch"],[5,2,1,"","_validate_imputed"],[5,2,1,"","_write_imputed_params_score"],[5,2,1,"","df2chunks"],[5,2,1,"","fit_predict"],[5,2,1,"","read_imputed"],[5,2,1,"","write_imputed"]],"pgsui.impute.iterative_imputer_fixedparams":[[6,1,1,"","IterativeImputerFixedParams"]],"pgsui.impute.iterative_imputer_fixedparams.IterativeImputerFixedParams":[[6,2,1,"","_impute_one_feature"],[6,2,1,"","_initial_imputation"],[6,4,1,"","clf_kwargs"],[6,4,1,"","clf_type"],[6,4,1,"","disable_progressbar"],[6,4,1,"","estimator"],[6,2,1,"","fit_transform"],[6,4,1,"","genotype_data"],[6,4,1,"","imputation_order"],[6,4,1,"","indicator_"],[6,4,1,"","initial_imputer_"],[6,4,1,"","initial_strategy"],[6,4,1,"","logfilepath"],[6,4,1,"","max_iter"],[6,4,1,"","max_value"],[6,4,1,"","min_value"],[6,4,1,"","n_features_with_missing_"],[6,4,1,"","n_iter_"],[6,4,1,"","n_nearest_features"],[6,4,1,"","pops"],[6,4,1,"","prefix"],[6,4,1,"","progress_update_percent"],[6,4,1,"","random_state_"],[6,4,1,"","sample_posterior"],[6,4,1,"","skip_complete"],[6,4,1,"","str_encodings"],[6,4,1,"","tol"],[6,4,1,"","verbose"]],"pgsui.impute.iterative_imputer_gridsearch":[[7,1,1,"","IterativeImputerGridSearch"]],"pgsui.impute.iterative_imputer_gridsearch.IterativeImputerGridSearch":[[7,2,1,"","_impute_one_feature"],[7,2,1,"","_initial_imputation"],[7,2,1,"","fit_transform"],[7,4,1,"","genotype_data"],[7,4,1,"","imputation_sequence_"],[7,4,1,"","indicator_"],[7,4,1,"","initial_imputer_"],[7,4,1,"","n_features_with_missing_"],[7,4,1,"","n_iter_"],[7,2,1,"","plot_search_space"],[7,4,1,"","random_state_"],[7,4,1,"","str_encodings"]],"pgsui.impute.neural_network_imputers":[[8,1,1,"","NeuralNetwork"],[8,1,1,"","UBP"],[8,1,1,"","VAE"]],"pgsui.impute.neural_network_imputers.NeuralNetwork":[[8,2,1,"","categorical_accuracy_masked"],[8,2,1,"","categorical_crossentropy_masked"],[8,2,1,"","categorical_mse_masked"],[8,2,1,"","fill"],[8,2,1,"","make_reconstruction_loss"],[8,2,1,"","masked_mse"],[8,2,1,"","mle"],[8,2,1,"","validate_batch_size"],[8,2,1,"","validate_extrakwargs"],[8,2,1,"","validate_input"]],"pgsui.impute.neural_network_imputers.UBP":[[8,2,1,"","_build_ubp"],[8,2,1,"","_create_missing_mask"],[8,2,1,"","_encode_onehot"],[8,2,1,"","_get_hidden_layer_sizes"],[8,2,1,"","_init_weights"],[8,2,1,"","_initialise_parameters"],[8,2,1,"","_remove_dir"],[8,2,1,"","_train_epoch"],[8,2,1,"","_train_on_batch"],[8,2,1,"","_validate_hidden_layers"],[8,2,1,"","fit"],[8,2,1,"","fit_transform"],[8,2,1,"","predict"],[8,2,1,"","reset_seeds"],[8,2,1,"","set_optimizer"]],"pgsui.impute.neural_network_imputers.VAE":[[8,2,1,"","_create_missing_mask"],[8,2,1,"","_create_model"],[8,2,1,"","_decode_onehot"],[8,2,1,"","_encode_categorical"],[8,2,1,"","_encode_onehot"],[8,2,1,"","_get_hidden_layer_sizes"],[8,2,1,"","_read_example_data"],[8,2,1,"","_train_epoch"],[8,2,1,"","fit"],[8,2,1,"","fit_transform"],[8,3,1,"","imputed"],[8,2,1,"","predict"]],"pgsui.impute.simple_imputers":[[9,1,1,"","ImputeAlleleFreq"],[9,1,1,"","ImputePhylo"]],"pgsui.impute.simple_imputers.ImputeAlleleFreq":[[9,2,1,"","fit_predict"],[9,2,1,"","write2file"]],"pgsui.impute.simple_imputers.ImputePhylo":[[9,2,1,"","allMissing"],[9,2,1,"","draw_imputed_position"],[9,2,1,"","get_internal_lik"],[9,2,1,"","get_iupac_full"],[9,2,1,"","get_nuc_colors"],[9,2,1,"","impute_phylo"],[9,2,1,"","is_int"],[9,2,1,"","label_bads"],[9,2,1,"","nbiallelic"],[9,2,1,"","parse_arguments"],[9,2,1,"","print_q"],[9,2,1,"","str2iupac"],[9,2,1,"","transition_probs"],[9,2,1,"","validate_arguments"]],"pgsui.read_input":[[11,0,0,"-","popmap_file"],[12,0,0,"-","read_input"]],"pgsui.read_input.popmap_file":[[11,1,1,"","ReadPopmap"]],"pgsui.read_input.popmap_file.ReadPopmap":[[11,2,1,"","__init__"],[11,2,1,"","read_popmap"],[11,2,1,"","validate_popmap"]],"pgsui.read_input.read_input":[[12,1,1,"","GenotypeData"],[12,5,1,"","merge_alleles"]],"pgsui.read_input.read_input.GenotypeData":[[12,2,1,"","blank_q_matrix"],[12,2,1,"","check_filetype"],[12,2,1,"","convert_012"],[12,2,1,"","convert_onehot"],[12,3,1,"","genotypes_df"],[12,3,1,"","genotypes_list"],[12,3,1,"","genotypes_nparray"],[12,3,1,"","genotypes_onehot"],[12,4,1,"","guidetree"],[12,3,1,"","indcount"],[12,3,1,"","individuals"],[12,4,1,"","num_inds"],[12,4,1,"","num_snps"],[12,4,1,"","onehot"],[12,2,1,"","parse_filetype"],[12,4,1,"","pops"],[12,3,1,"","populations"],[12,2,1,"","q_from_file"],[12,2,1,"","q_from_iqtree"],[12,2,1,"","read_phylip"],[12,2,1,"","read_phylip_tree_imputation"],[12,2,1,"","read_popmap"],[12,2,1,"","read_structure"],[12,2,1,"","read_tree"],[12,4,1,"","samples"],[12,3,1,"","snpcount"],[12,4,1,"","snps"]],"pgsui.utils":[[14,0,0,"-","misc"],[15,0,0,"-","sequence_tools"]],"pgsui.utils.misc":[[14,1,1,"","HiddenPrints"],[14,1,1,"","StreamToLogger"],[14,5,1,"","all_zero"],[14,5,1,"","generate_012_genotypes"],[14,5,1,"","generate_random_dataset"],[14,5,1,"","get_indices"],[14,5,1,"","get_processor_name"],[14,5,1,"","isnotebook"],[14,5,1,"","progressbar"],[14,5,1,"","timer"],[14,1,1,"","tqdm_linux"],[14,5,1,"","weighted_draw"]],"pgsui.utils.misc.StreamToLogger":[[14,2,1,"","flush"],[14,2,1,"","write"]],"pgsui.utils.misc.tqdm_linux":[[14,2,1,"","status_printer"]],"pgsui.utils.sequence_tools":[[15,5,1,"","blacklist_maf"],[15,5,1,"","blacklist_missing"],[15,5,1,"","countSlidingWindow"],[15,5,1,"","count_alleles"],[15,5,1,"","expand012"],[15,5,1,"","expandAmbiquousDNA"],[15,5,1,"","expandLoci"],[15,5,1,"","gc_content"],[15,5,1,"","gc_counts"],[15,5,1,"","getFlankCounts"],[15,5,1,"","get_iupac_caseless"],[15,5,1,"","get_iupac_full"],[15,5,1,"","get_major_allele"],[15,5,1,"","get_revComp_caseless"],[15,5,1,"","listToSortUniqueString"],[15,5,1,"","mask_content"],[15,5,1,"","mask_counts"],[15,5,1,"","n_lower_chars"],[15,5,1,"","remove_items"],[15,5,1,"","reverseComplement"],[15,5,1,"","seqCounter"],[15,5,1,"","seqCounterSimple"],[15,5,1,"","seqSlidingWindow"],[15,5,1,"","seqSlidingWindowString"],[15,5,1,"","simplifySeq"],[15,1,1,"","slidingWindowGenerator"],[15,5,1,"","stringSubstitute"]],pgsui:[[3,0,0,"-","impute"],[10,0,0,"-","read_input"],[13,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","property","Python property"],"4":["py","attribute","Python attribute"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:property","4":"py:attribute","5":"py:function"},terms:{"0":[1,4,5,6,7,8,9,12,14,15,16],"0002":4,"001":[4,6,7,8,14],"01":[4,8,16],"012":[4,5,8,9,12,14],"02":4,"06":4,"1":[1,4,5,6,7,8,9,12,14,15,16],"10":[4,6,7,16],"100":[4,5,8],"1000":[4,8],"11":16,"110":16,"15":[4,5,14],"196":4,"1960":[6,7],"1d":5,"1e":[4,6,7,8],"2":[1,4,5,6,7,8,9,12,14,15,16],"20":[1,4,14,16],"200":16,"200000":4,"2005":4,"2011":[6,7],"2016":4,"21":[1,4,6,7],"215":4,"22":[6,7],"23":[6,7],"24":1,"25":[5,7,16],"256":8,"2d":[4,5,9,12,16],"2nd":12,"3":[1,4,6,7,8,9,14,16],"30":[4,16],"300":4,"302":[6,7],"306":[6,7],"31":4,"32":[4,8],"35":14,"3887":4,"3895":4,"3d":12,"4":[4,5,6,7,9,12,14,16],"40":5,"45":[6,7],"5":[1,4,6,7,8,12,14,16],"50":[4,5,8,16],"5000":4,"57":1,"6":[1,4,6,7],"60":14,"6000":[6,7],"64":16,"67":[6,7],"6x":1,"6y5g_mh":1,"7":[1,6,7],"7hg3hcq2":1,"7x":4,"8":[1,4,5,14],"9":[1,4,5,6,7,8,9,12],"95":[4,5],"9584":[6,7],"9999":[6,7],"boolean":[5,8],"break":[5,16],"case":[4,8,9,15,16],"char":[9,15],"class":[4,5,6,7,8,9,11,12,14,15,16],"default":[1,4,5,6,7,8,9,12,14,15,16],"do":[1,4,5,7,8,12,15,16],"final":[1,4,5,6,7,16],"float":[4,5,6,7,8,9,12,14],"function":[4,5,6,7,8,9,12,14,15,16],"import":[1,6,7,8],"int":[4,5,6,7,8,9,12,14,15,16],"long":[4,16],"new":[1,4,5,14],"return":[4,5,6,7,8,9,11,12,14,15],"static":14,"super":16,"true":[4,5,6,7,8,9,12,14,16],"try":[9,16],"var":[1,4,15],"while":[4,8,15],A:[4,6,7,9,12],And:16,As:1,At:7,But:16,For:[4,6,7,9,16],If:[1,4,5,6,7,8,9,12,14,16],In:[4,9,16],It:[1,4,7,9,16],NO:12,NOT:7,Nearness:[4,6,7],No:[12,16],Not:[9,12],One:[4,6,8,12],Or:16,The:[1,4,5,6,7,8,9,12,16],Then:16,There:[1,4,16],These:16,To:[4,6,7,16],Will:[4,5,6,7,8,14,16],_:14,__:14,__file__:1,__init__:11,_average_list_of_dict:5,_build_ubp:8,_create_missing_mask:8,_create_model:8,_decode_onehot:8,_defile_dataset:5,_define_iterative_imput:5,_encode_categor:8,_encode_onehot:8,_estim:[6,7],_format_featur:5,_gather_impute_set:5,_get_best_param:5,_get_hidden_layer_s:8,_impute_df:5,_impute_ev:5,_impute_gridsearch:5,_impute_one_featur:[6,7],_impute_singl:5,_imputed_012:[4,9],_imputer_valid:5,_init_weight:8,_initial_imput:[6,7],_initialise_paramet:8,_io:14,_iter:[6,7],_mode_list_of_dict:5,_print_scor:5,_read_example_data:8,_remove_dir:8,_remove_invalid_col:5,_remove_nonbiallel:5,_subset_data_for_gridsearch:5,_train_epoch:8,_train_on_batch:8,_validate_hidden_lay:8,_validate_imput:5,_write_imputed_params_scor:5,ab:[6,7],abl:5,about:[6,7],abov:16,absolut:[4,5,6,7],accept:4,access:16,accord:[4,9],account:[6,7],accuraci:[4,5,7,16],across:[4,5,12,16],activ:[1,4,8,16],activationfunction1:8,activationfunctionn:8,actual:5,ad:[4,6,7,12,16],adaboost:4,adagrad:[4,8],adam:[4,8],add:[6,7,8,14],add_ind:[6,7],addit:[4,9],adjust:[1,4,8,16],affect:4,after:[4,5,6,7,8,9,16],algorithm:[1,4,5,7,16],align:[9,12],all:[1,4,5,6,7,8,9,12,14,15,16],all_list:15,all_zero:14,allel:[4,5,9,12,14,15],allmiss:9,allow:[4,5,6,7,8,14,16],almost:5,aln:12,alnfil:9,alpha:4,alpha_1:4,alpha_2:4,alpha_init:4,also:[1,4,6,7,8,9,14,16],alt:[4,9],altern:[4,12,14,16],altogeth:4,alwai:[4,5],ambigu:15,among:5,amount:4,an:[1,4,5,6,7,8,12,14,15,16],analys:16,analysi:[4,8,16],ancestr:9,ani:[1,4,5,6,7,8,9,14,16],anoth:15,anywher:5,api:[6,7,16],appear:[6,7],append:16,appli:[4,6,7,8],approach:4,appropri:[4,9,12],approxim:4,ar:[1,4,5,6,7,8,9,12,15,16],arab:[4,6,7],arbitrari:4,architectur:[],arg:4,argment:[],argument:[4,5,6,7,8,9,16],argv:1,arm64:1,arm:[],around:[4,9],arr1:5,arr2:5,arr:16,arrai:[4,5,6,7,8,9,12,14,16],ascend:[4,6,7],assert:5,assertionerror:[5,9,12],assess:[5,16],assum:[4,9,12,15],asterisk:9,astyp:8,attempt:7,attribut:12,attributeerror:6,auto:4,autoencod:[4,5,8,16],automat:[1,16],avail:[1,4,7,16],averag:[5,16],awai:4,ax:[],axi:[],axis1:[],axis2:[],axis3:[],axisgrid:7,background:[],backpropag:[4,8,16],bad:[5,9],bad_list:15,bag:4,balanc:7,ball_tre:4,balltre:4,bar:[0,4,5,6,7,8,9,14],base:[1,4,5,6,7,8,9,11,12,14,15,16],bash:1,basic:[4,7,8],batch:[4,8],batch_siz:[4,8,16],bayesian:4,bayesian_ridge_imp_default:[],bayesianridg:[4,6,7],bbox:[],bbox_to_anchor:[],becaus:[4,7],been:[1,4,6,7,8],befor:[4,5,6,7],behind:[],being:[4,6,7,9,16],below:[1,9],best:[4,5,8,16],best_param:[4,5],best_scor:5,better:[4,16],between:[4,5,6,7,14,16],bi:[5,9,12],bia:4,biallel:5,bin:[1,4],binari:[6,7],bioinformat:4,biolog:16,bit:5,black:[],blacklist_maf:15,blacklist_miss:15,blank_q_matrix:12,bool:[4,5,6,7,8,9,12,14,15],boost:[4,16],booster:4,boosting_typ:4,bootstrap:4,border:[],borderaxespad:[],both:[4,5,6,7,8,9,16],box:[],branch:4,broadcast:[6,7],brute:4,buck:[6,7],buf:14,build:4,buuren:[6,7],by_popul:[4,9,16],c:[1,4,6,7,9,12,15],calcul:[4,8,9,16],call:[4,5,6,7,9,12,16],callabl:[4,5,6,7,8],callback:[4,5,7],calucal:8,can:[1,4,5,6,7,8,9,14,16],cannot:[4,9],carriag:14,categor:[8,16],categorical_accuracy_mask:8,categorical_crossentropy_mask:8,categorical_mse_mask:8,ceil:4,chain:[6,7],chang:[1,5,6,7,16],channel:1,charact:[14,15],check:[5,9,12,14],check_filetyp:12,checkpoint:16,child:[4,9],chip:1,choos:[4,16],chunk:[0,5],chunk_siz:[4,5,16],ci:4,circl:[],clade:9,classif:4,classifi:[4,5,6,7,16],clf:5,clf_kwarg:[5,6,7],clf_random_st:4,clf_tol:4,clf_type:[5,6,7],clone:[6,7],close:1,closer:4,cluster:[4,6,7,8,14,16],code:[1,4,9,15],coeffici:[4,6,7],col_selection_r:5,color:9,cols_to_keep:[5,6,7],colsample_bytre:4,column:[4,5,6,7,8,9,12,14,16],column_perc:5,column_subset:[4,5,9,16],columns_to_subset:5,columnspac:[],com:1,combin:16,command:1,common:[5,8,15],commonli:[4,8],compar:5,comparison:[5,16],compat:[4,5,6,7,12],compil:[1,8],complement:15,complet:[1,16],complete_encod:8,compon:[4,8,16],comput:[1,4,6,7,8,9,16],concaten:5,conda:1,condit:[4,6,7,8],confid:[4,5],conjuct:[],conjunct:9,consecut:[4,5,7],consensu:15,consid:[4,16],consist:[12,14],constant:6,construct:4,constructor:11,consum:4,contain:[4,5,6,7,9,12,14],content:15,context:16,continu:16,contour:7,contrast:4,contribut:4,control:[4,6,7,14],converg:[4,6,16],convergencewarn:6,convert:[5,6,7,8,9,12],convert_012:12,convert_onehot:12,copi:1,core:[5,9,12,16],correct:[4,5,8,9,12],correctli:[4,8,16],correl:[4,6,7,16],correspod:5,correspond:[5,6,7,8,16],cost:4,could:12,count:[5,15],count_allel:15,countslidingwindow:15,cover:[],coverag:[4,6,7],cpu:[1,4,16],creat:[1,8,15],criteria:[4,6],criterion:[4,5,6,7,8,16],cross:[4,5,7,8],crossentropi:8,crossov:4,crossover_prob:4,csv:[4,5,9],cumul:[],current:[1,4,5,6,7,8,9,12,16],custom:[5,8,16],cv:[4,5,7,8,16],cwd:1,cycl:[6,7,8,16],d:[8,14],dart:4,data:[0,4,5,6,7,8,9,12,14],datafram:[4,5,6,7,8,9,12,16],datapoint:[6,7],dataset:[4,5,8,12,14,16],deap:[1,4],deap_1d32f65d60a44056bd7031f3aad44571:1,debug:[4,6,7],decemb:1,decis:[4,16],decod:8,decompos:[4,9],decomposit:[],decor:14,decreas:[4,8],defaul:9,defin:[4,5,8,9,16],delta:4,dens:8,denselayer1:8,denselayer2:8,denselayer3:8,denselayer4:8,denselayern:8,denseoutputlay:8,densiti:7,depend:[0,4,16],deprec:[6,7],depth:4,descend:[4,6,7,9,15],descent:[4,8],desend:9,design:16,despit:[6,7],detail:4,detect:1,determin:[4,5,6,7,9,16],devianc:4,deviat:8,df2chunk:5,df:[4,5,9,16],df_chunk:5,df_dummi:8,df_score:5,dict:[4,5,6,7,8,9,11,12,14,15],dictionari:[4,5,6,7,8,9,11,12,16],did:[4,6,7],differ:[4,5,8,12,16],dim1:8,dim2:8,dim_reduction_supported_algorithm:[],dim_reduction_supported_argu:[],dimens:[4,8,16],dimension:[4,8,16],dimreduction_plot_set:[],diploid:[4,9,15],diploidi:15,dir_path:8,directli:[4,8],directori:8,dirti:5,disabl:[4,5,6,7,8,9,16],disable_progressbar:[4,5,6,7,8,9,16],discard:[6,7],disk:[5,8,9,11,12,16],displai:[6,7],dist:15,distanc:[4,9],distancemetr:4,distribut:[4,5,7,8,16],diverg:[],divid:[8,12],document:[4,12,16],doe:[1,4,5,6,7,8,9,12,16],doesn:[5,8],don:[5,16],done:[4,5],down:16,downgrad:1,download:1,draw:[4,9],draw_imputed_posit:9,drawn:[4,6,7],drop:5,dropout1:8,dropout2:8,dropout3:8,dropout:[4,8],dropout_prob:[4,8],dtype:[6,7],dure:[1,4,5,6,7,8,9],dynam:14,e:[4,5,8,12,15,16],eacah:4,each:[4,5,6,7,8,9,14,16],eamupluslambda:4,earli:[4,5,6,7,8,16],earlier:16,early_stop_gen:[4,5,7,8],edg:[],edgecolor:[],effect:4,efficaci:16,egg:1,egg_info:1,either:[4,5,6,7,8,9,16],electron:[6,7],element:15,elit:4,els:1,elu:[4,8,16],embed:[],empti:[5,14],enabl:4,enable_iterative_imput:[6,7],encod:[4,5,6,7,8,9,12,14],encodings_dict:12,encount:1,end:16,ensembl:[4,5],ensur:[4,6,7,8],entri:[],entropi:4,env:1,environ:[1,16],epoch:[4,8,16],equal:[4,8],equat:[6,7],equationspackag:[6,7],equival:4,error:[4,5,8,16],especi:4,estim:[0,2,3,5,6,7,8],eta:4,etc:12,etim:4,euclidean:4,evalu:[4,6,7,8],evalut:[5,16],even:[6,7,12],everi:[4,5,6,7,14,16],evolutionari:4,exactli:8,exampl:[5,6,7,8,14],except:[7,16],exclude_n:9,exec:1,exist:[1,8,9],exit:1,expan:15,expand012:15,expand:[4,8,9,15],expandambiquousdna:15,expandloci:15,expect:[5,6,7],expens:16,experi:16,experiment:[6,7],explicitli:[6,7],explor:5,exponenti:4,extend:16,extens:4,extra:[4,8,16],extra_tre:4,extract:5,extratre:4,extratreesclassifi:4,extrem:[4,16],f:[1,4,8],facecolor:[],facet:7,facilit:16,fact:16,factor:4,fail:7,fairli:4,fake:14,fals:[4,5,6,7,8,9,12,14,15,16],fanci:16,far:[4,16],fashion:[4,6,7],fast:15,faster:[4,5,15],feat_idx:[6,7],featur:[4,5,6,7,8,12,16],few:16,fewer:7,fewest:[4,6,7],figheight:[],figwidth:[],file:[4,5,6,7,8,9,11,12,14],filenam:[4,5,9,11,12,16],filenotfounderror:12,filetyp:[9,12,16],fill:8,find:[4,5,16],fine:1,first:[1,4,8,12,16],fit:[4,5,6,7,8],fit_mod:[6,7],fit_predict:[4,5,9],fit_transform:[6,7,8],fix:16,flag:[4,6,7],flank:15,fluctuat:16,flush:14,fname:12,fold:[4,5,7,16],folder:1,follow:[1,5,8,16],font:[],forest:[4,16],forg:1,forgo:1,format:[4,5,6,7,8,9,12,14,15,16],forward:8,found:[5,7,9,12],four:12,fp_write:14,fraction:[4,5],frame:[5,9,12],framealpha:[],frequenc:[4,9],friedman:4,friedman_ms:4,from:[1,4,5,6,7,8,9,11,12,14,15,16],full:[4,5,12],fulli:[],func:14,further:4,futur:12,g:[4,5,6,7,8,9,12,15,16],ga:[4,5,7,16],ga_algorithm:4,ga_kwarg:[5,7],gain:4,gamma:4,gap:15,gasearchcv:[4,5,7],gashler:4,gather:5,gaussian:[6,7],gb:16,gbdt:4,gblinear:4,gbtree:4,gc:15,gc_content:15,gc_count:15,gener:[4,5,6,7,9,14,15,16],generate_012_genotyp:14,generate_random_dataset:14,genet:[1,4,5,7,16],geno:15,genotyp:[4,5,6,7,8,9,12,14,15],genotype_data:[4,6,7,8,9,16],genotypedata:[4,6,7,8,9,12],genotypes_df:[12,16],genotypes_list:[12,16],genotypes_nparrai:[12,16],genotypes_onehot:12,gentoyp:7,get:[4,5,7,8,9,15,16],get_indic:14,get_internal_lik:9,get_iupac_caseless:15,get_iupac_ful:[9,15],get_major_allel:15,get_nuc_color:9,get_processor_nam:14,get_revcomp_caseless:15,get_weight:8,getattr:1,getflankcount:15,gini:4,github:1,give:[14,15],given:[4,5],global:[4,9,16],global_af:16,glorot_norm:[4,8],good:16,goss:4,gradient:[4,8,16],gradient_boosting_imp_default:[],gradientboostingclassifi:4,gradienttap:8,greater:[4,8],grid:[1,4,5,7],grid_cv:7,grid_it:4,grid_n_it:7,grid_n_job:7,grid_param:16,gridparam:[4,5],groothui:[6,7],grow:4,gt:[4,8,9],gui:4,guid:[6,7,9,12],guidetre:[6,9,12,16],ha:[1,4,6,7,8,12,16],had:[9,16],halt:[4,8],handl:7,happen:16,have:[1,4,5,6,7,9,12,16],header:12,height:7,help:16,here:[1,4,5,6,7,16],herein:[4,6,7,16],hessian:4,het:[4,9],heterogyg:12,heterozyg:[4,9],heterozygot:[12,14],hex:9,hh:14,hidden:[4,8,16],hidden_activ:[4,8,16],hidden_layer_s:[4,8,16],hiddenlayern:8,hiddenprint:14,high:[4,8,16],higher:[4,6,7],highest:8,hl_func:8,hom:[4,9],homozyg:[4,9],hot:[8,12],how:[6,7,15,16],howev:4,hpc:[4,6,7,14],html:4,http:[1,4],huge:[4,6,7],hundr:16,hyper:4,hyperparamet:7,i:[4,12,16],id:[4,5,6,7,9,12,16],ignor:[5,6,7,8,9],ilgbm:[],imp_kwarg:5,imp_mean:[6,7],implement:[4,7,8,16],improv:[4,5,6,7,8,16],impur:4,imput:[0,2,12],imputation_ord:[4,6,7],imputation_sequence_:7,impute_mod:12,impute_phylo:9,imputeallelefreq:[4,5,6,9,16],imputebayesianridg:4,imputegradientboost:[4,16],imputeknn:[4,16],imputelgbm:16,imputelightgbm:[4,16],imputemf:[4,16],imputenlpca:[4,16],imputephylo:[4,5,9,16],imputerandomforest:[4,16],imputeubp:[4,16],imputeva:[4,16],imputexgboost:[4,16],in_fit:[6,7],includ:[4,6,7,8,9,12,16],increas:[4,16],ind:16,indcount:[12,16],index:[0,5,6,7,8,9,12],indexerror:9,indic:[5,6,7,9,12,14],indicator_:[6,7],indid:11,individu:[4,12,16],induc:4,induct:[6,7],inf:[6,7],influenc:4,info:[1,4,8],inform:[4,6,7,8,9,16],inherit:[],initi:[0,4,5,6,7,8,9],initial_imputer_:[6,7],initial_strategi:[4,5,6,7,8,16],inner:12,input:[0,4,5,6,7,8,9,12,14],input_data:8,inputlay:8,insert:[9,14],insid:[],instal:0,instanc:[4,5,6,7,9,14],instanti:[4,8,16],instead:[1,4,5,6,8,9,16],instruct:[],integ:[4,5,6,7,8,9,12,14,16],intel:[1,4],intelex:[1,4],intellig:4,intend:[4,9,16],interact:4,intern:[4,9],interv:[4,5],intract:4,introduc:5,invalid:5,invers:[4,8],involv:[4,16],io:1,ioerror:12,iq:[9,12,16],iqfil:12,iqtre:[9,12,16],is_int:[5,9],isn:1,isnotebook:14,issu:[4,6,7],item:[8,15],iter:[0,4,5,6,7,14,15],iterative_imputer_fixedparam:[2,3,5],iterative_imputer_gridsearch:[2,3,5],iterative_mod:[4,9],iterativeimput:[4,5,6,7,9,16],iterativeimputerfixedparam:[5,6],iterativeimputergridsearch:[5,7],its:[4,6,7,8,9,16],iupac:[6,7,9,12,15],j:4,job:[4,5],journal:[6,7],jupyt:14,jupyterlab:1,just:[4,5,6,7,8,9,16],k:[4,5,16],kaplan:4,karin:[6,7],kd_tree:4,kdtree:4,keep:[5,6,7],kei:[4,5,6,7,8,9,12,16],kera:[1,4,8,16],kernel_initi:[4,8],keyword:[4,5,6,7,8,9,16],knn:16,knn_imp_defaults_it:[],knn_imp_defaults_nonit:[],knnimput:[],know:[4,5,16],known:[4,5,8,16],known_val:[6,7],kopka:4,kwarg:[4,5,8,9],l1:[4,8],l1_penalti:[4,8],l2:[4,8],l2_penalti:[4,8],l:[4,5,14,15,16],l_p:4,label:[5,9,12,16],label_bad:9,labelspac:[],lack:[7,16],lambda:[4,8],lambda_1:4,lambda_2:4,lambda_init:4,larg:4,last:1,latent_featur:4,latest:1,layer:[4,8,16],lead:[4,16],leaf:[4,16],leaf_siz:4,learn:[1,4,5,6,7,8,16],learner:4,learning_r:[4,8,16],least:[4,7],leav:[4,9,16],left:[4,5,6,7],legend:[],legend_edgecolor:[],legend_insid:[],legend_loc:[],len:5,length:[4,7,8,9,12,16],less:[4,5,6,7,8],let:1,level:[4,6,8,16],lgbm:16,librari:4,light:4,lightgbm:[1,4,16],lik_arr:9,like:[6,7,8,14,16],likelihood:[8,9],likewis:16,limit:4,line:[1,12,14],linear:[4,8,16],linear_model:[],linspac:16,linux:14,list:[4,5,6,7,8,9,12,14,15,16],listtosortuniquestr:15,load:[],loc:15,local:16,locat:[],loci:[4,5,9,15,16],locu:[4,9],log2:[4,8,16],log:[5,6,7],log_level:14,logarithm:16,logfil:6,logfilepath:[5,6,7],logger:14,logist:4,loguniform:4,longer:14,look:[4,16],loop:[7,8],loss:[4,8],lot:[9,15,16],low:[4,16],lower:[4,5,15,16],lowest:16,m1:1,m:4,mac:[],machin:16,macosx:1,magnitud:4,mai:[1,4,12,14],maintain:4,make:[4,7,8,9,15,16],make_reconstruction_loss:8,manag:14,manhattan_dist:4,mani:[4,6,7,15,16],map:[4,8,9,11,12,16],marker:7,markeredgecolor:[],markeredgewidth:[],markerfirst:[],markers:[],markerscal:[],martinez:4,mask:[6,7,8,15],mask_cont:15,mask_count:15,mask_missing_valu:[6,7],masked_ms:8,match:[12,14],math:4,matplotlib:1,matrix:[4,6,7,8,9,12],max:[6,7],max_alt_r:14,max_delta_step:4,max_depth:[4,16],max_epoch:[4,8],max_featur:[4,16],max_het_r:14,max_it:[4,5,6,7,16],max_leaf_nod:4,max_missing_r:14,max_sampl:4,max_valu:[6,7,14],maxim:[4,16],maximum:[4,5,6,7,8,9,14],maxiumum:[8,14],maxk:[],md:[],mds_default_set:[],mean:[4,5,6,8,16],mean_squared_error:5,measur:[4,6,7],median:[4,5,6],medium:[],meet:4,member:14,memori:4,merg:12,merge_allel:12,messag:[4,6,7],met:[4,6,7,8],method:[0,4,6,7,8],metric:[4,5,7,16],mf:16,mice:[6,7],midpoint:[4,8,16],might:[4,6,7,16],min:[6,7],min_alt_r:14,min_child_sampl:4,min_child_weight:4,min_het_r:14,min_impurity_decreas:4,min_missing_r:14,min_samples_leaf:[4,16],min_samples_split:[4,16],min_split_gain:4,min_valu:[6,7,14],min_weight_fraction_leaf:4,miniconda3:1,miniforg:1,miniforge3:1,minim:4,minimmum:16,minimum:[4,5,6,7,14,16],minkowski:4,minkowski_dist:4,misc:[2,13],miss:[4,5,6,7,8,9,12,14,16],missing:[6,7],missing_mask:8,missing_row_mask:[6,7],missing_v:5,missing_valu:[6,7,8],missingind:[6,7],misss:14,ml:0,mle:8,mlflow:1,mlp:8,mm:14,mode:[4,5,6,7,8,12,14,16],model:[4,5,6,7,8,16],model_evalu:4,model_mlp_phase2:8,model_mlp_phase3:8,model_single_lay:8,modif:16,modifi:7,modul:[0,1],more:[4,6,7,8,16],morri:4,most:[1,4,5,6,7,8,12,15],most_frequ:[4,6,7,8,16],mostli:1,mputelgbm:[],mse:4,mtrand:4,multi:[4,8],multidimension:[],multipl:[4,6,7],multivari:[6,7],mushroom:8,must:[4,5,6,7,8,9,12,16],mutat:4,mutation_prob:4,n:[1,4,6,7,9,15],n_batch:8,n_categori:8,n_compon:[4,8,16],n_dim:8,n_estim:[4,5,16],n_featur:[4,5,6,7,8],n_features_in_chunk:5,n_features_with_missing_:[6,7],n_iter:[4,5],n_iter_:[6,7],n_job:[4,5,16],n_lower_char:15,n_nearest_featur:[4,6,7,16],n_neighbor:4,n_sampl:[4,5,6,7,8,12,16],n_site:12,n_snp:[12,16],n_step:4,na:[6,7],name:[5,11,12,14],nan:[5,6,7,8],natur:[4,16],nbiallel:9,ncol:14,ndarrai:[4,5,6,7,8,9,12,14],nearest:[0,4,6,7],necessari:[1,8,16],necessarili:[4,6,7],need:[1,4,6,7,16],neg:[6,9],neg_root_mean_squared_error:4,neighbor:[0,4,6,7],neighbor_feat_idx:[6,7],neighborhood:4,neightbor:7,network:[0,4,8],neural:[0,4,8],neural_network_imput:[2,3],neuralnetwork:8,neuron:[4,8],newick:[9,12,16],newli:4,newlin:14,next:4,nlpca:[4,8,16],nn:[],node:[4,9,16],node_index:9,nois:4,non:[0,4,5,6,8,12,14,15],none:[4,5,6,7,8,9,12,15,16],nonetyp:[4,5,9],nonlinear:16,nopop:16,normal:[1,6,7,8],note:[1,4,6,7,14,16],notebook:14,now:[1,6,7],np:[5,6,7,8,16],nrow:14,ns:15,nuc:9,nucleotid:[4,6,7,9,12,15],nullabl:[6,7],num:[15,16],num_class:8,num_hidden_lay:[4,8,16],num_ind:[12,16],num_leav:4,num_sampl:14,num_snp:[12,16],number:[4,5,6,7,8,9,12,14,15],numer:[5,16],numpi:[1,4,5,6,7,8,9,12,14,16],o:[],object:[4,5,6,7,8,9,11,12,14,15,16],observ:[4,9],obtain:[4,5,9,16],occur:[1,4,6,7,8,16],occurr:[6,7,16],off:[4,16],often:[6,7,16],onc:[4,5,6,7,8],one:[1,4,5,6,7,8,9,12,15,16],onehot:12,onerow:12,ones:5,onli:[1,4,5,6,7,8,9,12,16],onto:[6,7],oob_scor:4,open:1,oper:4,opposit:12,opt:[1,7,16],optim:[4,6,8],optimum:4,option:[4,5,6,7,8,9,12,14,15,16],order:[4,5,6,7,9,12,15],org:4,origin:[4,5,8],original_num_col:5,os:1,oserror:12,other:[1,4,6,7,8,16],otherwis:[4,5,6,7,8,9,12,14,16],oudshoorn:[6,7],our:16,out:[1,4,5,9,15,16],out_typ:8,outlin:9,output:[1,4,5,6,7,8,9,12,14,16],output_activ:[4,8],output_format:[4,9],outputlay:8,over:[4,8,16],overal:[4,8],overfit:[4,8,16],overlai:[],overload:5,overridden:[6,7,14],own:[9,16],p:[4,9],packag:1,pad:[],page:0,pair:[4,6,7],pairgrid:7,palett:[],panda:[1,4,5,6,7,8,9,12,16],paralel:16,parallel:[0,4,5],paramet:[1,4,5,6,7,8,9,11,12,14,15,16],params_list:5,parent:9,parenthes:5,pars:[11,12],parse_argu:9,parse_filetyp:12,parsimoni:9,partit:4,pass:[4,8,16],past:[1,4,8],patch:4,path:[1,5,6,7,9,12,14,16],pattern:16,pca:[4,16],pca_cumvar_default_set:[],pca_default_set:[],pd:[6,7,9],pdf:[5,9,16],penalti:[4,8],per:[4,8,9,12,14,16],percent:[5,6],percentag:[6,7],perceptron:[4,8],perform:[4,5,6,7,8,16],permut:4,pg:[1,16],pgsui:[2,3,10,13],phase:[4,6,7,8],phy:16,phylip:[6,7,9,12],phylipfil:9,phylo:16,phylogenet:[4,9,12,16],phylogeni:[4,6,7,8,9],pick:4,pip:1,place:14,placehold:[6,7],plot:[5,7,9,16],plot_search_spac:7,po:[9,15],point:[4,8],pop:[4,5,6,7,9,12,16],pop_af:16,popid:[11,12],popmap:[11,12],popmap_fil:[2,10],popmap_filenam:11,popmapfil:[9,12,16],popul:[4,5,6,7,8,9,11,12,16],population1:16,population2:16,population_s:4,posit:[6,12],possibl:[4,6,7,8,9,12,14,15],posterior:[6,7],postord:9,power:4,precis:4,precomput:4,predict:[4,5,6,7,8],prefix:[4,5,6,7,8,9,14,16],preorder:9,present:[7,12],preserv:15,prevent:16,primari:15,primarili:[4,5,9],princip:[4,8],print:[4,5,6,7,9,14,16],print_q:9,prior:[4,7],prioriti:[4,9],privat:1,probabilist:4,probabl:[4,6,7,9],problem:[1,4],proce:1,procedur:4,process:[0,1,4,6,7,9],processor:[1,4,5,7,16],progress:[0,4,5,6,7,8,9,14],progress_update_frequ:6,progress_update_perc:[4,5,6,7,16],progressbar:14,properti:[4,8,12],proport:[4,5,6,7,8,14,15,16],provid:[4,5,6,7,8,9,12,15,16],pseudo:[4,6,7],pt:9,pull:16,pure:4,purpos:[4,9],py:1,pyplot:[],python3:1,python:[1,16],q:[4,8,9,12,16],q_from_fil:12,q_from_iqtre:12,qmatrix:[6,7,9,12,16],qmatrix_iqtre:[9,12,16],qualiti:4,queri:4,quick:[4,5],r:[1,4,6,7],rais:[4,5,6,8,9,12],ram:[4,6,16],random:[4,5,6,7,8,14,16],random_forest_imp_default:[],random_forest_unsupervised_default:[],random_forest_unsupervised_support:[],random_st:[4,6,7],random_state_:[6,7],randomforestclassifi:[4,5],randomizedsearchcv:[4,5,7,16],randomli:[4,5,14],randomsearchcv:[5,7],randomst:[4,6,7],randomtreesembed:[],rang:4,rate:[4,8,9,12,14,16],ratio:4,rcparam:[],re:1,reach:[4,6,7,8],read:[1,4,5,6,7,8,9,11,12],read_imput:5,read_input:[0,2,4,9],read_phylip:12,read_phylip_tree_imput:12,read_popmap:[11,12],read_structur:12,read_tre:12,readpopmap:11,recent:[6,7],recommend:16,record:8,recov:4,recurr:[4,8],recurrent_weight:[4,8],redirect:14,reduc:[4,8,16],reduct:4,ref:[4,9,15],refer:[4,6,7,12,14],refin:[4,8,16],refit:[6,7],refresh:14,reg_alpha:4,reg_lambda:4,region:15,regress:[4,6,7],regressor:[4,5,6],regular:[4,8],regularization_param:4,rel:4,releas:1,relev:16,reload:[4,8],relu:[4,8,16],remain:[5,9,12],remov:[5,6,7,8,15],remove_item:15,rep:5,replac:[1,4,5,14,15],replic:[4,5,8,16],report:16,repres:12,reproduc:4,request:14,requir:[0,4,6,7,8,9,16],reset:8,reset_se:8,resourc:[4,16],respect:5,result:4,retain:9,retriev:16,return_std:[6,7],revers:15,reversecompl:15,rf:[4,16],ridg:4,right:[4,6,7],rmse:5,robin:[6,7],robust:4,roman:[4,6,7],root:[4,8,9,16],round:[4,6,7],row:[5,8,12,14,16],royal:[6,7],rule:16,run:[1,4,5,6,7,8,12,16],runtim:14,s:[1,4,5,6,7,14,15,16],same:[4,6,7,8,9,12,16],sampl:[4,5,6,7,8,9,11,12,15,16],sample1:16,sample2:16,sample3:16,sample4:16,sample_posterior:[6,7],sample_weight:4,sampleid:[5,6,7,9,12],save:[4,5,6,7,8,9,16],save_plot:[9,16],scalar:[6,7],scale:4,scatter:7,scholz:4,scikit:[1,4,5,6,7,16],scipi:[1,4],score:[4,5,7],scoring_metr:[4,5,7],screen:[6,7,16],seaborn:[1,7],search:[0,1,4,5,7],search_spac:[5,7],second:[4,8,9,12,16],see:[4,6,7,8,12,16],seed:[4,6,7,8],selbig:4,select:[4,5,6,7],self:[5,6,7,12,14],sensibl:4,separ:[9,16],sepcifi:[4,9],seq:15,seqcount:15,seqcountersimpl:15,seqslidingwindow:15,seqslidingwindowstr:15,sequenc:[4,9,12,15],sequence_tool:[2,13],sequenti:8,serv:4,set1:[],set:[4,5,6,7,8,9,12,16],set_optim:8,setup:1,setuptool:1,sever:[6,7,16],sgd:[4,8],sh:1,shadow:[],shape:[4,5,6,7,8,9,12,16],shift:15,should:[1,4,5,6,7,8,9,12,16],show:7,shrink:4,side:4,sigmoid:[4,8],signific:[4,6,7],silent:4,similar:[4,16],simpl:[4,7],simple_imput:[2,3],simpleimput:[5,6,7],simplifi:15,simplifyseq:15,sinc:[1,5,6,7,8],singl:[4,5,6,7,8,15],site:[4,5,7,8,9,12,16],site_r:9,size:[0,4,5,7,8,14],skip:[1,6],skip_complet:[4,6,7],sklearn:[1,4,5,6,7,16],sklearn_genet:[4,7],slice:15,slide:15,slidingwindowgener:15,slow:16,slowli:[4,8,16],small:4,smaller:4,smith:4,smooth:4,snp:[4,9,12,15,16],snp_data:12,snp_index:9,snpcount:[12,16],snpsdict:9,so:[1,4,8,9,16],societi:[6,7],softmax:8,softwar:[6,7],solut:4,some:[1,4,5,9,16],sometim:[4,8],sort:[5,15],sourc:[4,5,6,7,8,9,11,12,14,15],space:[4,5,7],span:4,specif:1,specifi:[4,5,8,9,12,16],speed:[1,4,6,7],split:[4,5,8,12,15,16],sqrt:[4,8,16],squar:[4,8,16],ss:14,stabl:4,stack:[6,7],stage:4,standalon:16,standard:[4,6,7,8,12,16],start:[],stat:4,state:[4,9],statement:14,statist:[5,6,7],statu:[1,4,5,6,7,9,14,16],status_print:14,std:14,stdout:14,stef:[6,7],step:[1,4,6,7,9],still:[4,5,6,7,9,16],stochast:4,stop:[4,5,6,7,8,16],store:[4,6,7],str2iupac:9,str:[4,5,6,7,8,9,11,12,14,15,16],str_encod:[4,6,7,9],strategi:[0,4,6,7,8],stratifi:[7,16],stream:14,streamtologg:14,string:[4,5,8,9,14,15,16],stringio:1,stringsubstitut:15,structur:[4,6,7,8,9,12,15],structure1row:[9,12,16],structure1rowpopid:[9,12],structure2row:[9,12,16],structure2rowpopid:[9,12],studi:16,style:[12,15],subsampl:4,subsample_for_bin:4,subsample_freq:4,subset:[4,5,7,9,16],subst:15,substitut:16,substr:15,success:1,sui:[1,16],suit:16,suitabl:[6,7],sum:4,supervis:16,suppli:[4,8,9,12,14,16],support:[0,1,4,5,6,7,8,12],supported_imputation_set:[],supress:14,surround:9,sy:1,system:16,t6g4kn711z5cxmc2_tvq0mlw0000gn:1,t:[1,4,5,6,7,8,9,12,16],tab:16,tabl:[9,16],take:[4,9,14,16],target:[4,6,7,8],tell:16,tensor:8,tensorflow:[1,8],term:4,termin:1,test:[1,4,5,6,7,8,14,16],textiowrapp:14,tf:[4,8],than:[4,5,6,7,8,14,15,16],thei:[4,9,16],them:[5,16],thi:[1,4,6,7,8,9,15,16],third:4,those:[4,5],thousand:16,three:[4,8],threshold:15,through:[1,4,5,7,8,16],throughout:[4,6,7],thumb:16,time:[4,5,6,7,14,16],timer:14,tip:[9,16],titl:[],title_fonts:[],togeth:5,token:1,tol:[4,6,7,8],toler:[4,6,7,8],too:[4,8,16],total:[4,15,16],total_loci:4,tournament:4,tournament_s:4,toyplot:9,toytre:[1,9,12],tqdm:[1,4,5,6,7,8,14,16],tqdm_linux:14,trade:4,tradit:4,train:[4,6,7,8,16],train_epoch:[4,8,16],transform:[4,6,7,9],transit:9,transition_prob:9,transpar:[],travers:9,tre:16,tree:[4,9,12,16],treefil:[6,7,9,12],troubleshoot:0,tsne_default_set:[],tune:[4,16],tupl:[5,6,7,9],turn:[4,16],tutori:0,two:[4,5,7,12,15,16],tyler:1,type:[4,5,6,7,8,9,11,12,14,15,16],typeerror:[4,5,8,9,12],typic:4,ubp:[4,8,16],undergo:16,uniform:4,union:[4,5,6,7,9,12],uniqu:[5,15,16],unit:[4,8],univari:[6,7],unknown:9,unlimit:4,unprovid:16,unsupervis:[4,8,16],unsupport:[8,9],until:[4,8],up:[1,4,5,6,7,16],updat:[4,5,6,7,9,14,16],upon:[6,7],upper:[4,5],us:[1,4,5,6,7,8,9,12,14,15,16],usag:[4,16],use_2to3:[],user:[1,4,6,7,14,16],usual:[4,8,9,16],utf:14,util:[0,2],v58:1,v:8,v_latent:8,vae:[4,5,8,16],val:9,valid:[0,4,5,6,7,8,9,12,15],valid_col:[6,7],validate_argu:9,validate_batch_s:8,validate_extrakwarg:8,validate_input:8,validate_popmap:11,validation_mod:5,validation_onli:[4,5,8,16],valu:[4,5,6,7,8,9,12,14,16],valueerror:[5,6,8,9,12],van:[6,7],variabl:[4,5,8],varianc:4,variat:[4,5,8,16],variou:[5,16],vcf:[12,15],vector:8,verbos:[4,6,7,9],veri:16,verifi:8,version:[1,7],versionad:6,versionchang:[6,7],versu:[6,7,16],vertic:[],via:5,visual:[],w:[4,8,14],w_mean:8,w_stddev:8,wa:[1,4,6,7],wai:[15,16],want:[4,8,16],watch:8,we:[1,4,5,6,7,8,16],weight:[4,8],weighted_draw:14,weights_initi:[4,8],well:4,were:[1,5,9,12],what:16,when:[4,6,7,9,16],where:[4,6,7,8],wherea:4,whether:[4,5,6,7,8,9,12,14,15],which:[1,4,5,6,7,8,9,16],white:[],whole:[4,5,16],width:15,window:15,wise:[4,9],within:14,without:[4,5,6,7,16],won:[6,7,16],work:[1,4,6,7,8,14,16],workaround:1,wors:16,worthwhil:4,would:[7,16],write2fil:9,write:[4,6,9,14,16],write_imput:5,write_output:[4,8,9,16],written:[5,16],x:[4,5,6,7,8,9,15,16],x_:[6,7],x_fill:[6,7],x_hat:8,x_missing_mask:[6,7],x_pred:8,x_t:[6,7],x_true:8,xgb:[4,16],xgboost:[1,4,16],xt:[6,7],xx:[],y:[4,6,7,8,15],y_pred:8,y_true:8,yet:12,you:[1,4,6,7,8,9,16],your:16,yourself:16,zero:14},titles:["Welcome to PG-SUI\u2019s documentation!","Installation","PG-SUI Modules","impute","pgsui.impute.estimators module","pgsui.impute.impute module","pgsui.impute.iterative_imputer_fixedparams module","pgsui.impute.iterative_imputer_gridsearch module","pgsui.impute.neural_network_imputers module","pgsui.impute.simple_imputers module","read_input","pgsui.read_input.popmap_file module","pgsui.read_input.read_input module","utils","pgsui.utils.misc module","pgsui.utils.sequence_tools module","Tutorial"],titleterms:{allel:16,architectur:1,arm:1,attribut:16,bar:16,chunk:16,content:[0,2,3,10,13],cross:16,data:16,depend:1,document:0,error:1,estim:[4,16],factor:16,file:16,frequenc:16,genotypedata:16,grid:16,gridparam:16,imput:[3,4,5,6,7,8,9,16],indic:0,initi:16,input:16,instal:1,invalid:1,iter:16,iterative_imputer_fixedparam:6,iterative_imputer_gridsearch:7,load:16,mac:1,matrix:16,maximum:16,method:16,misc:14,ml:16,modul:[2,3,4,5,6,7,8,9,10,11,12,13,14,15],n:16,nearest:16,neighbor:16,network:16,neural:16,neural_network_imput:8,non:16,number:16,parallel:16,pg:[0,2],pgsui:[4,5,6,7,8,9,11,12,14,15],phylip:16,phylogeni:16,popmap:16,popmap_fil:11,process:16,progress:16,quick:[],read_input:[10,11,12],requir:1,s:0,search:16,sequence_tool:15,set:[],simple_imput:9,size:16,start:[],strategi:16,structur:16,sui:[0,2],support:16,tabl:0,troubleshoot:1,tutori:16,use_2to3:1,util:[13,14,15],valid:16,welcom:0}})