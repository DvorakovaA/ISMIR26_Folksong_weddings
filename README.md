# ISMIR26_Folksong_weddings
Repository for supplementary material regarding ISMIR 2026 publication Towards Cross-Cultural Study of Folksong Lyrics with Machine Translation

## Data processing
Collecting those with lyrics  
Translating  
Dropping refused translation  
-> results in `translation/translated`  
  
Subset of songs (mainly wedding ones) were selected into the human MT evaluation (see `translation\eval_translation`).  
  
For experiments with classification by language (to classify typology labels) only songs of 10 most frequent labels (that are not `other` or `unknown` etc) were used (which can be found in `experiments\topic_models\selected_ids` and `data_selection\README.md`).  

For experiments with classification of wedding songs by language of origin 
we took all the wedding subset (5 languages) and then also only sets of Czech, Dutch and Ukrainian as 
those with better translation results in the human evaluation.

Some additional results of MT and also classification are present altogether in `RESULTS.md`.

## Machine translation
### Translation
openai/gpt-4.1-mini  

### Evaluation
Results of human evaluation of the MT can be found in `translation\eval_translation\results`.


## Topic modeling
BERTopic  

Files with topic distribution obtained from pretrained `MaartenGr/BERTopic_Wikipedia`
model are in `experiments\topic_models\bertopic_output` parquet files.  
  
Files with topic distribution obtained from training BERTopic models on folk lyrics data are in `experiments\topic_models\bertopic_output` in model description named files (same names as in `classification` folder).
  


Results of classification experiments can be found in `experiments\topic_models\classification_output`. Mainly interesting are subfolders with suffix `subsample_fifth`, which contains results using subsampling reducing the largest classes to the size of the fifth largest one.

For all experiments (5 by language and two versions of the wedding subset) we have thre kinds of results:
- on the selected set as it is (unbalanced classes)
- with subsampling (we sort classes by frequency and look for the largest gap to find a cutoff, all bigger are subsampled to the size of the class closest smaller to the gap)
- with subsampling reducing the largest classes to the size of the fifth largest one  

Second set of experiments is using training of BERTopic model on our own data (`train_bertopic` folder). Data used for training were subsampled - particulary Korean set size was reduced approx. to Estonian. Folders with results reflect models parametrization (e.g. `wed_bt_mts_10_topics_30` means 30 topics with minimal topic size 10), missing topics number specification means that "auto" was used.
  
**Replicability instructions**:  
1. `bertopic_train.py` - train BERTopic architecture model given data (when own data are used)
2. `compare_lables.py` - create a hard assignment overview
3. `bertopic_pipeline.py` - assign topic probabilities to given data using pretrained model (local or name from HF)
4. `transform_to_tb.py` - optional step to create datafiles (metadata.tsv and topics.tsv) to be used in tensorboard projector
5. `all_feat_select.sh` (runs `feature_select_pipeline.py` on all language pairs)
6. `classification.sh` - run classification experiment (SVM, RBF) on given set of labels and probabilities

  
## Wedding subset
IDs of songs that were used in experiments related to wedding (resp. marriage) songs are in `wedding_set` folder.