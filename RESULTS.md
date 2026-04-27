# Results that did not fit into the paper

Here we present an additional results that did not fit into the paper page limit.  


## MACHINE TRANSLATION

Relation between the overall rating and correctness for each evaluated song. (Axis y does not start at 0.)
![Relation between the overall rating and correctness for each evaluated song. (Axis y does not start at 0.)](translation/eval_translation/results/correctness_vs_rating_all_languages.png)

-------------------------------

Overview of the relation between the annotated level of unusual language used and correctness for each evaluated song. Axes y does not start at 0.
![Overview of the relation between the annotated level of unusual language used and correctness for each evaluated song. (Axes y does not start at 0.)](translation/eval_translation/results/correctness_vs_unusualLanguage_all_languages.png)



## LYRICS VS. TYPOLOGY
### Classification: monolingual corpora
#### CS
Embeddings of CS songs in 3D PCA space -> we see mainly mess.
![Embeddings of CS songs in 3D PCA space](experiments/topic_models/bertopic_output/tensorflow_projection/cs_tb/cs_pca.png)

-------------------------------------
Confusion matrix with counts

-------------------------------------
-------------------------------------
#### NL
-------------------------
Embeddings of NL songs in 3D PCA space.  
Green is St.Nicholas, dark blue is anti-marriage, pink is children, orange is wedding.  
![Embeddings of NL songs in 3D PCA space. ](experiments/topic_models/bertopic_output/tensorflow_projection/nl_tb/nl_pca_1.2.3.lables.png)


### Cross-Cultural Classification: Wedding songs

-------------
Embeddings of wedding songs in 3D PCA space.  
Purple: KO, light blue: NL, dark blue: CS, green: UK, red: ET.
![Embeddings of wedding songs in 3D PCA space. Purple: KO, light blue: NL, dark blue: CS, green: UK, red: ET.](experiments/topic_models/bertopic_output/tensorflow_projection/weddings_tb/weddings_pca_lang_labels.png)


------------------------
#### Confusion matrices

**Without subsampling**
![](experiments/topic_models/classification_output/weddings/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

**With gap subsmapling**
![](experiments/topic_models/classification_output/weddings_subsample/RBF_SVM_cv_aggreg_matrix_normal_wedd_subsample.png)

**With fifth subsampling**
![](experiments/topic_models/classification_output/weddings_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)


