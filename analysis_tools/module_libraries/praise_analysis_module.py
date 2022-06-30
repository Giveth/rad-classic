import pandas as pd
import numpy as np
import plotly.express as px

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
nltk.download('stopwords')

# cleaning master function
def clean_praise(praise):
    # code adapted from: https://ourcodingclub.github.io/tutorials/topic-modelling-python/
    my_stopwords = nltk.corpus.stopwords.words('english')
    word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem # clean words to the "stem" (e.g. words->word, talked->talk)
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

    praise = praise.lower() # lower case
    praise = re.sub('['+my_punctuation + ']+', ' ', praise) # strip punctuation
    praise = re.sub('\s+', ' ', praise) #remove double spacing
    praise = re.sub('([0-9]+)', '', praise) # remove numbers
    praise_token_list = [word for word in praise.split(' ')
                            if word not in my_stopwords] # remove stopwords

    praise_token_list = [word_rooter(word) if '#' not in word else word
                        for word in praise_token_list] # apply word rooter

    praise = ' '.join(praise_token_list)
    return praise


def data_by_quantifier(praise_data):
    quant_only = pd.DataFrame()
    #praise_data.drop(['DATE', 'TO USER ACCOUNT', 'TO USER ACCOUNT ID', 'TO ETH ADDRESS', 'FROM USER ACCOUNT', 'FROM USER ACCOUNT ID', 'FROM ETH ADDRESS', 'REASON', 'SOURCE ID', 'SOURCE NAME', 'AVG SCORE'], axis=1, inplace=True)
    num_of_quants = len(praise_data.filter(like='QUANTIFIER').columns)
    for i in range(num_of_quants):
        q_name =  str( 'QUANTIFIER '+ str(i+1) +' USERNAME' )
        q_value = str('SCORE '+str(i+1) )
        q_duplicate = str('DUPLICATE ID '+str(i+1) )
        buf = praise_data[['ID', q_name , q_value, q_duplicate]].copy()

        #delete the duplicated rows
        buf = buf.loc[buf[q_duplicate].isnull()] # only include the non-duplicated rows
        buf = buf[['ID', q_name , q_value]] # don't need the duplication info anymore
        buf=buf.dropna() # NA comes from when the quantifier is less than given in cross-period analysis      
    
        buf.rename(columns={q_name: 'QUANT_ID',  q_value: 'QUANT_VALUE', 'ID':'PRAISE_ID'}, inplace=True)

        quant_only = quant_only.append(buf.copy(), ignore_index=True)

    columnsTitles = ['QUANT_ID', 'PRAISE_ID', 'QUANT_VALUE']
    quant_only.sort_values(['QUANT_ID', 'PRAISE_ID'], inplace=True)
    quant_only =  quant_only.reindex(columns=columnsTitles).reset_index(drop=True)
    return quant_only


class praise_quantifier():
    def __init__(self, praisedata=pd.DataFrame(), quantifiertable=pd.DataFrame()):
        self.df = praisedata
        
        if len(quantifiertable)==0:
            quantifiertable = data_by_quantifier(praisedata.copy())
        self.quantratingdf = quantifiertable
        self.quantlist = self._get_valid_quantifier()
        self.metrics_df = self._get_participant_metrics()
        return
    

    def _get_valid_quantifier(self):
        df = self.df
        quantifier_rating_table = self.quantratingdf
        quant_cols = df.filter(like='USERNAME', axis=1)
        quantlist = np.unique(quant_cols.values)
        valid_qt = np.ones_like(quantlist, dtype=bool)*True
        for kq, quantid in enumerate(quantlist):
            quantdf = quantifier_rating_table.loc[quantifier_rating_table['QUANT_ID'] == quantid]
            if set(quantdf['QUANT_VALUE'].values) == {0}:
                valid_qt[kq] = False
            if quantid == 'None':
                valid_qt[kq] = False
        quantlist = quantlist[valid_qt]
        return quantlist

    def get_outlier(self, keywords, maxscore=5):
        from re import search  # for searching sub strings

        """
        keywords is a string in regex pattern. 
        For example, to single out all the praise about attending meetings:
        keywords = 'attend|showing up|join'
        """
        quantifier_rating_table = self.quantratingdf
        df = self.df
        for QUANT_ID in self.quantlist:
            quantdf = quantifier_rating_table.loc[quantifier_rating_table['QUANT_ID'] == QUANT_ID]
            av_scores = []
            for kr, row in quantdf.iterrows():
                praise_id = row['PRAISE_ID']
                praise_reason = df.loc[df['ID'] ==
                                       praise_id]['REASON'].values[0]
                # if multiple things are mentioned in this praise, let's skip it.
                if search(keywords, praise_reason) and ('and' not in praise_reason):
                    prase_avscore = df.loc[df['ID'] ==
                                           praise_id]['AVG SCORE'].values[0]
                    quantifier_score = row['QUANT_VALUE']
                    # when others most likely don't give it such high score
                    if quantifier_score > maxscore and prase_avscore <= maxscore:
                        print(
                            f'{QUANT_ID} gave {quantifier_score} for the praise "{praise_reason}"')

    def _get_participant_metrics(self):
        # TODO: make this more modularized so the user can easily choose what analysis to include
        quantifier_coef = []
        quantifier_score_displace = []

        quantifier_rating_table = self.quantratingdf
        df = self.df
        for QUANT_ID in self.quantlist:
            quantdf = quantifier_rating_table.loc[quantifier_rating_table['QUANT_ID'] == QUANT_ID]
            av_scores_others = []
            score_quant = []
            scoredist = []
            scoredisplace = []
            for kr, row in quantdf.iterrows():
                quantifier_score = row['QUANT_VALUE']
                praise_id = row['PRAISE_ID']
                praise_row = df.loc[df['ID'] == praise_id]

                otherscores = praise_row.filter(
                    like='SCORE ').values.tolist()[0]
                otherscores.remove(quantifier_score)
                
                otherscores = np.array(otherscores)
                otherscores = otherscores[~np.isnan(otherscores)]

                if len(otherscores) <2:
                    continue
                av_scores_others.append(np.mean(otherscores))
                score_quant.append(quantifier_score)
                # like "standar deviation" from this score
                scoredist.append(
                    np.sqrt(sum((quantifier_score - otherscores)**2))/(len(otherscores)))
                scoredisplace.append(np.mean(quantifier_score - otherscores))
               
                
            coef = np.corrcoef(
                score_quant, av_scores_others)[1, 0]
            quantifier_coef.append(coef)
            quantifier_score_displace.append(np.mean(scoredisplace))
        quantifier_metrics_df = pd.DataFrame(index=self.quantlist, data={
                                             'pearson_coef': quantifier_coef, 'av_score_displacement': quantifier_score_displace})
        return quantifier_metrics_df

    def plot_coefficient(self):
        """returns a plotly express figure object"""
        quantifier_metrics_df = self.metrics_df
        fig = px.scatter(quantifier_metrics_df.sort_values(by='pearson_coef'),
                         y='pearson_coef', title="How similiar is one's score with the other quantifiers?")
        return fig

    def plot_mean_displacement(self):
        """returns a plotly express figure object"""
        quantifier_metrics_df = self.metrics_df
        fig = px.bar(quantifier_metrics_df.sort_values(by='av_score_displacement'), y='av_score_displacement',
                     title="Do one tends to give higher or lower scores than the other quantifiers?")
        return fig



# courtesy of @inventandchill
# n_senders: Left side. Praise senders. n largest ones + rest (others)
# n_receivers: Right side. Praise receivers. n largest ones + rest (others)


def prepare_praise_flow(dataframe_in, n_senders, n_receivers):
    reference_df = dataframe_in[['FROM', 'TO', 'AVG SCORE']].copy()
    reference_df.reset_index(inplace=True, drop=True)
    reference_df.dropna(subset=['FROM', 'TO', 'AVG SCORE'], inplace=True)
    reference_df.reset_index(inplace=True, drop=True)

    # Left side. Praise senders. X largest ones + rest (others). (-1 because of zero-counting)
    n1 = n_senders - 1
    # Right side. Praise receivers. Y larget one + rest (others) (-1 because of zero-counting)
    n2 = n_receivers - 1

    df_from = reference_df.groupby(['FROM']).sum().copy()
    df_from.reset_index(inplace=True, drop=False)
    min_from = df_from['AVG SCORE'].sort_values(ascending=False).unique()[n1]
    df_from2 = df_from.copy()
    df_from2.loc[df_from2['AVG SCORE'] < min_from, 'FROM'] = 'Rest from 1'

    df_to = reference_df.groupby(['TO']).sum().copy()
    df_to.reset_index(inplace=True, drop=False)
    min_to = df_to['AVG SCORE'].sort_values(ascending=False).unique()[n2]
    df_to2 = df_to.copy()
    df_to2.loc[df_to2['AVG SCORE'] < min_to, 'TO'] = 'Rest to 1'

    df3 = reference_df.copy()
    i = 0

    length_data = df3.shape[0]

    while (i < length_data):
        if (not(df3.at[i, 'FROM'] in df_from2['FROM'].unique())):
            df3.at[i, 'FROM'] = 'REST FROM'
        if (not(df3.at[i, 'TO'] in df_to2['TO'].unique())):
            df3.at[i, 'TO'] = 'REST TO'

        i = i+1

    df4 = df3.copy()

    df4 = df4.groupby(['FROM', 'TO']).sum().copy()

    df4.reset_index(inplace=True, drop=False)
    df4['TO'] = df4['TO']+' '

    return df4


def noShow(a, b):
    if int(a) == 0 and bool(b) == False:
        return np.nan
    else:
        return a


def spread_sort(praise_distribution, NUMBER_OF_QUANTIFIERS_PER_PRAISE):
    # for general use
    col_dismissed = [
        f'DISMISSED {k+1}' for k in range(NUMBER_OF_QUANTIFIERS_PER_PRAISE)]
    col_dupids = [
        f'DUPLICATE ID {k+1}' for k in range(NUMBER_OF_QUANTIFIERS_PER_PRAISE)]
    col_scores = [
        f'SCORE {k+1}' for k in range(NUMBER_OF_QUANTIFIERS_PER_PRAISE)]

    # clean the Dataframe and remove praise where there is dismissal agreement
    praisecheck_df = praise_distribution.drop(['TO ETH ADDRESS', 'TO USER ACCOUNT ID', 'FROM ETH ADDRESS',
                                              'FROM USER ACCOUNT ID', 'SOURCE ID', 'SOURCE NAME', 'PERCENTAGE', 'TOKEN TO RECEIVE'], axis=1)
    ethadds = [
        f'QUANTIFIER {k+1} ETH ADDRESS' for k in range(NUMBER_OF_QUANTIFIERS_PER_PRAISE)]
    praisecheck_df.drop(ethadds, axis=1, inplace=True)

    praisecheck_clean_controversial = praisecheck_df.loc[praisecheck_df[col_dismissed].sum(
        axis=1) < NUMBER_OF_QUANTIFIERS_PER_PRAISE, :]

    # reinstate the original scores for the duplicates before calculating the spread
    dupclean_praisecheck = praisecheck_clean_controversial.copy()
    for i, row in praisecheck_clean_controversial.iterrows():
        for j, dup_id_label in enumerate(col_dupids):

            if row[dup_id_label] is not None:
                # find the score
                find_value = praisecheck_df.loc[praisecheck_df['ID']
                                                == row[dup_id_label], col_scores[j]]

                try:
                    # substitute it in dupclean
                    dupclean_praisecheck.at[i, str(
                        col_scores[j])] = int(find_value)
                except:
                    # account for the bug in early rounds
                    dupclean_praisecheck.at[i, str(col_scores[j])] = 0

                    # discard no-shows (score = 0 and not dismissed, after the above check for duplicates) and calculate spread

    for i,  score_col in enumerate(col_scores):
        dupclean_praisecheck[score_col] = dupclean_praisecheck.apply(
            lambda x: noShow(x[score_col], x[col_dismissed[i]]), axis=1)

    dupclean_praisecheck['SPREAD'] = dupclean_praisecheck[col_scores].max(
        axis=1) - dupclean_praisecheck[col_scores].min(axis=1)
    sort_by_controversial = dupclean_praisecheck.sort_values(
        by='SPREAD', ascending=False).reset_index()
    return sort_by_controversial
