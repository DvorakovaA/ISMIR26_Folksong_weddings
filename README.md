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


### Topic modeling
BERTopic  

Files with topic distribution obtained from pretrained `MaartenGr/BERTopic_Wikipedia`
model are in `experiments\topic_models\bertopic_output`.  


Results of classification experiments can be found in `experiments\topic_models\classification_output`. Mainly interesting are subfolders with suffix `subsample_fifth`, which contains results using subsampling reducing the largest classes to the size of the fifth largest one.

For all experiments (5 by language and two versions of the wedding subset) we have thre kinds of results:
- on the selected set as it is (unbalanced classes)
- with subsampling (we sort classes by frequency and look for the largest gap to find a cutoff, all bigger are subsampled to the size of the class closest smaller to the gap)
- with subsampling reducing the largest classes to the size of the fifth largest one


## Wedding subset
IDs of songs that were used in experiments related to wedding (resp. marriage) songs are in `wedding_set` folder.