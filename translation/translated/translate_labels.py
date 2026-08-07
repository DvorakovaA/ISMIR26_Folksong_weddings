"""
Script that translates labels in estonian and czech sets.
"""
import pandas as pd

LABEL_TRANSLATION = {
    "cs": {'SVATEBNÍ' : 'wedding',
           'MILOSTNÉ' : 'love',
           'Písně o stavích, živnostech a jiných stránkách života občanského' : 'bussiness_and_civil',
           'Písně věku mládeneckého a panenského' : 'young_people',
           'ŽERTOVNÉ A TANEČNÍ' : 'entertainment_and_dance',
           'DĚTSKÉ' : 'children',
           'KOLEDY' :'carols',
           'VOJENSKÉ' : 'military',
           'Písně společenské' : 'social_life',
           'Písně a říkadla výroční' : 'calendar'
    },
    "et" : 
        {'kalendrilaulud' : 'calendar',
            'lastelaulud' : 'children',
            'laulud_abielust' : 'marriage',
            'laulud_kodust_ja_lapsepolvest' : 'home_and_childhood',
            'laulud_meelelahutamiseks' : 'entertainment',
            'laulud_noorrahva_elust' : 'youth_life',
            'laulud_uhiskondlikest_vahekordadest' : 'songs about social relationships',
            'looduslaulud' : 'nature',
            'murelaulud' : 'worry',
            'toolaulud' : 'work',

            #'Pulmalaulud ' : 'wedding'
    }
}

"""
    "et": {'Kalendrilaulud ' : 'calendar',
            'Lastelaulud ' : 'children',
            'Laulud abielust ' : 'marriage',
            'Laulud kodust ja lapsepõlvest ' : 'home_and_childhood',
            'Laulud meelelahutamiseks ' : 'entertainment',
            'Laulud noorrahva elust ' : 'youth_life',
            'Laulud ühiskondlikest vahekordadest ' : 'songs about social relationships',
            'Looduslaulud ' : 'nature',
            'Murelaulud ' : 'worry',
            'Töölaulud ' : 'work',
            #'Pulmalaulud ' : 'wedding'
    }
"""

LANG_FILES = {
    #"cs": "cs_translated.csv",
    "et": "et_translated.csv"
}

def main():
    for lang, file in LANG_FILES.items():
        df = pd.read_csv(file)
        print(len(df), "rows in", file)
        print(LABEL_TRANSLATION[lang])
        df["label"] = df["label"].map(LABEL_TRANSLATION[lang]) #df["thematic_group"].map(LABEL_TRANSLATION[lang])
        df['thematic_group'] = df['thematic_group'].str.strip()

        print(len(df))

        print(df['label'].value_counts())
        print(len(df['label'].dropna()), "rows with translated labels in", file) 
        #print(df['thematic_group'].value_counts())
        # drop unlabeled rows
        df.dropna(subset=['label'], inplace=True)
        # save ids
        df.to_csv(file, index=False)
        print(len(df), "rows with translated labels in", file)
        
        print(len(df['item_id']), "rows with translated labels in", file)
        df['item_id'].to_csv(f"{lang}_ids.csv", index=False)
        

if __name__ == "__main__":    
    main()