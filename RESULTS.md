# Results that did not fit into the paper

Here we present an additional results that did not fit into the paper page limit.  


# MACHINE TRANSLATION

Relation between the overall rating and correctness for each evaluated song. (Axis y does not start at 0.)
![Relation between the overall rating and correctness for each evaluated song. (Axis y does not start at 0.)](translation/eval_translation/results/correctness_vs_rating_all_languages.png)

-------------------------------

Overview of the relation between the annotated level of unusual language used and correctness for each evaluated song. Axes y does not start at 0.
![Overview of the relation between the annotated level of unusual language used and correctness for each evaluated song. (Axes y does not start at 0.)](translation/eval_translation/results/correctness_vs_unusualLanguage_all_languages.png)



# LYRICS VS. TYPOLOGY
## Classification: monolingual corpora
### CS
Embeddings of CS songs in 3D PCA space -> we see mainly mess.
![Embeddings of CS songs in 3D PCA space](experiments/topic_models/bertopic_output/tensorflow_projection/cs_tb/cs_pca.png)

-------------------------------------
#### Without subsampling
![](experiments/topic_models/classification_output/cs/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.2691,431.0
Linear_SVM,0.1308,431.0
```

-------------------------------------
#### With gap subsampling
![](experiments/topic_models/classification_output/cs_subsample/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.2587,419.0
Linear_SVM,0.1597,419.0
```

-------------------------------------
#### With fifth subsampling
![](experiments/topic_models/classification_output/cs_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

And also the same results with absolute (aggregated) counts while keeping recall color scale.
![](experiments/topic_models/classification_output/cs_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_counts_recall.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.2568,411.4
Linear_SVM,0.2347,411.4
```

-------------------------------------
#### CS using original labels without merging on same data selected by default picked merging
Showing one fold...

![](experiments/topic_models/classification_output/cs_orig_labels/RBF_SVM_cv_fold_confusion_matrix_counts_fold_0.png)


-------------------------------------
#### CS using strategy with more label merging and then taking top 10 labels of this different merge
![](experiments/topic_models/classification_output/cs_new_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.2734,420.4
Linear_SVM,0.2149,420.4
```

-------------------------------------
-------------------------------------
### NL
-------------------------
Embeddings of NL songs in 3D PCA space.  
Green is St.Nicholas, dark blue is anti-marriage, pink is children, orange is wedding.  
![Embeddings of NL songs in 3D PCA space. ](experiments/topic_models/bertopic_output/tensorflow_projection/nl_tb/nl_pca_1.2.3.lables.png)

-------------------------------------
#### Without subsampling
![](experiments/topic_models/classification_output/nl/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.7067,447.6
Linear_SVM,0.4741,447.6
```

-------------------------------------
#### With gap subsampling
![](experiments/topic_models/classification_output/nl_subsample/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.6592,377.2
Linear_SVM,0.5609,377.2
```

-------------------------------------
#### With fifth subsampling
![](experiments/topic_models/classification_output/nl_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

And also the same results with absolute (aggregated) counts while keeping recall color scale.
![](experiments/topic_models/classification_output/nl_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_counts_recall.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.5349,230.6
Linear_SVM,0.5105,230.6
```

-------------------------------------
-------------------------------------
### ET
-------------------------------------

#### Without subsampling
![](experiments/topic_models/classification_output/et/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.5253,472.4
Linear_SVM,0.3952,472.4
```

-------------------------------------
#### With gap subsampling
![](experiments/topic_models/classification_output/et_subsample/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.5248,470.2
Linear_SVM,0.4112,470.2
```

-------------------------------------
#### With fifth subsampling
![](experiments/topic_models/classification_output/et_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

And also the same results with absolute (aggregated) counts while keeping recall color scale.
![](experiments/topic_models/classification_output/et_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_counts_recall.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.5131,461.2
Linear_SVM,0.4584,461.2
```


-------------------------------------
-------------------------------------
### KO
-------------------------------------
#### Without subsampling
![](experiments/topic_models/classification_output/ko/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.6668,548.0
Linear_SVM,0.4827,548.0
```

-------------------------------------
#### With gap subsampling
![](experiments/topic_models/classification_output/ko_subsample/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.6376,494.0
Linear_SVM,0.6218,494.0
```

-------------------------------------
#### With fifth subsampling
![](experiments/topic_models/classification_output/ko_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

And also the same results with absolute (aggregated) counts while keeping recall color scale.
![](experiments/topic_models/classification_output/ko_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_counts_recall.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.6317,488.8
Linear_SVM,0.6196,488.8
```


-------------------------------------
-------------------------------------
### UK
-------------------------------------
#### Without subsampling
![](experiments/topic_models/classification_output/uk/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.5187,300.0
Linear_SVM,0.4363,300.0
```

-------------------------------------
#### With gap subsampling
![](experiments/topic_models/classification_output/uk_subsample/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.4557,236.4
Linear_SVM,0.4363,236.4

```

-------------------------------------
#### With fifth subsampling
![](experiments/topic_models/classification_output/uk_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)

And also the same results with absolute (aggregated) counts while keeping recall color scale.
![](experiments/topic_models/classification_output/uk_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_counts_recall.png)


```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.5196,293.6
Linear_SVM,0.4644,293.6
```

-------------------------------------
## Cross-Cultural Classification: Wedding songs

-------------
Embeddings of wedding songs in 3D PCA space.  
Purple: KO, light blue: NL, dark blue: CS, green: UK, red: ET.
![Embeddings of wedding songs in 3D PCA space. Purple: KO, light blue: NL, dark blue: CS, green: UK, red: ET.](experiments/topic_models/bertopic_output/tensorflow_projection/weddings_tb/weddings_pca_lang_labels.png)


------------------------
### All five languages

**Without subsampling**
![](experiments/topic_models/classification_output/weddings/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)
  
```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.692,303.0
Linear_SVM,0.6095,303.0
```

**With gap subsmapling**
![](experiments/topic_models/classification_output/weddings_subsample/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)
  
```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.6986,284.0
Linear_SVM,,0.6383,284.0
```

**With fifth subsampling**
![](experiments/topic_models/classification_output/weddings_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)
  

And also the same results with absolute (aggregated) counts while keeping recall color scale.
![](experiments/topic_models/classification_output/weddings_subsample_fifth/RBF_SVM_cv_aggregate_confusion_matrix_counts_recall.png)

```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.6309,215.6
Linear_SVM,0.6305,215.6
```
--------------------
### Only CS, NL, and UK

**Without subsampling**
![](experiments/topic_models/classification_output/high_weddings/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)
  
```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.6728,203.2
Linear_SVM,0.6708,203.2
```

**With gap subsmapling**
![](experiments/topic_models/classification_output/high_weddings_subsample/RBF_SVM_cv_aggregate_confusion_matrix_normalised.png)
  
```
model,cv_weighted_f1_mean,cv_n_pca_components_mean
RBF_SVM,0.6454,186.2
Linear_SVM,0.6984,186.2
```

Fifth subsampling does not make sense as we have only 3 classes.

