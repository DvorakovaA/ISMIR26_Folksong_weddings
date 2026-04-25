# Corpora processing

## Sources
CS: [https://pisnovna.cz/sbirky/o-sbirce?q=E](https://pisnovna.cz/sbirky/o-sbirce?q=E)  
UK: [https://github.com/aljanaki/Symbolic_Corpus_Of_Ukrainian_Folk_Music/](https://github.com/aljanaki/Symbolic_Corpus_Of_Ukrainian_Folk_Music/)  
ET: [https://www.folklore.ee/regilaul/andmebaas/?ln=en](https://www.folklore.ee/regilaul/andmebaas/?ln=en)  
NL: [https://www.liederenbank.nl/index.php?lan=en](https://www.liederenbank.nl/index.php?lan=en)  
KO: [https://github.com/danbinaerinHan/finding-tori/](https://github.com/danbinaerinHan/finding-tori/)
  

Following procedures were applied to obtain texts:
### CS
Erben (Prostonárodní české písně a říkadla) + Homolka Praha + Homolka Podřipko + Thořová Havlíčkobrodsko   
(Texts in form of .txt files were obtained from EI CAS.)


### NL
Scraped (with permission) using keywords search of following search **keywords**:  
 bruiloftslied, begrafenislied, verjaardagslied, ambachtslied, zeemanslied, soldatenlied, wiegelied, schommellied, driekoningenlied, sinterklaaslied, kinderlied, anti-huwelijkslied  
  
Selected songs from 18th, 19th and 20th century collections.


### ET
**Filtration applied: ** 
Region: Järvamaa  
Upper groups:  
Looduslaulud, Kalendrilaulud, 
Laulud meelelahutamiseks, 
Töölaulud, 
Murelaulud, 
Laulud ühiskondlikest vahekordadest, 
Laulud sõjast ja nekrutist, 
Laulud kodust ja lapsepolvest, 
Laulud noorrahva elust, 
Pulmalaulud, 
Laulud abielust, 
Lastelaulud,
Loitsud
  
-> XML export for each group  
  
After that, songs which were found in exports for more than one upper group were discarded.

### UK and KO
Whole distributed set from respective GitHubs


## Distribution of typological labels in sets selected for classification experiments
As for the classification experiment we picked only songs with ten most frequent labels.

**NL:**  
|label | count |
|-------|----------:|
|children         | 1491  |
|wedding          | 487  |
|work_sea         | 188  |
|work            |  134  |
|work_soldier    |   78 | 
|birthday        |   74  |
|st_nicholas     |   51  |
|anti-marriage   |   47  |
|lullaby         |   38  |
|christmas      |    24  |

**CS:**
|label | count |
|-------|----------:|
| young_people  | 794 |
| entertainment_and_dance  | 414 |
| business_and_civil  |  385 |
| love |  370 |
| calendar  | 310 |
| wedding | 204 |
| social_life | 129 |
| children | 88 |
| military     | 86 |
| carols | 49 |


**KO:**
|label | count |
|-------|----------:|
|Work_agricultural     |    3663 |
|Work_artisan          |     725 |
|Lullaby_lullaby       |     668 |
|Ritual_funeral        |     587 |
|Lament_lament         |     579 |
|Work_construction     |     463 |
|Work_fishing          |     317 |
|Lament_eosayong       |     293 |
|etc_minstrel          |     245 |
|Entertainment_children |    227 |

**UK:**
|label | count |
|-------|----------:|
|Non-ritual_Ballad            | 300 |
|Calendar-ritual_spring       | 274 |
|Family-ritual_wedding        | 257 |
|Calendar-ritual_winter       | 217 |
|Calendar-ritual_Summer       | 191 |
|Non-ritual_Lyrical           | 155 |
|Non-ritual_Humorous          |  83 |
|children_children           |   43 |
|Non-ritual_Social-domestic |    38 |
|Non-ritual_Romance        |     27 |


**ET:**
|label | count |
|-------|----------:|
|youth_life              |  715|
|entertainment            |  533|
| social_relationships |  400|
| work                            |  398|
| nature                         |  378|
| calendar                     |  303|
| marriage                     |  231|
| home_and_childhood       |  208|
| worry                          |  206|
| children                      |  181|