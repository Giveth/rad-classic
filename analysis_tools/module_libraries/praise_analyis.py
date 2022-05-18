import pandas as pd
import numpy as np
import plotly.express as px

class praise_round():
    def __init__(self,praisedata = pd.DataFrame(),quantifiertable = pd.DataFrame()):
        self.df= praisedata
        self.quantratingdf = quantifiertable
        self.quantlist = self._get_valid_quantifier()
        self.metrics_df = self._get_participant_metrics()
        return

    def _get_valid_quantifier(self):
        df= self.df
        quantifier_rating_table=self.quantratingdf
        quant_cols = df.filter(like='USERNAME',axis=1)
        quantlist = np.unique(quant_cols.values)
        valid_qt = np.ones_like(quantlist,dtype=bool)*True
        for kq,quantid in enumerate(quantlist):
            quantdf = quantifier_rating_table.loc[quantifier_rating_table['QUANT_ID']==quantid]
            if set(quantdf['QUANT_VALUE'].values)=={0}:
                valid_qt[kq]=False
        quantlist = quantlist[valid_qt]
        return quantlist
    
    def get_outlier(self,keywords,maxscore=5):
        from re import search # for searching sub strings

        """
        keywords is a string in regex pattern. 
        For example, to single out all the praise about attending meetings:
        keywords = 'attend|showing up|join'
        """        
        quantifier_rating_table=self.quantratingdf
        df = self.df
        for QUANT_ID in self.quantlist:
            quantdf = quantifier_rating_table.loc[quantifier_rating_table['QUANT_ID']==QUANT_ID]
            av_scores = []
            for kr, row in quantdf.iterrows():
                praise_id = row['PRAISE_ID']
                praise_reason = df.loc[df['ID']==praise_id]['REASON'].values[0]
                if search(keywords, praise_reason) and ('and' not in praise_reason): # if multiple things are mentioned in this praise, let's skip it.
                    prase_avscore = df.loc[df['ID']==praise_id]['AVG SCORE'].values[0]
                    quantifier_score=row['QUANT_VALUE']
                    if quantifier_score>maxscore and prase_avscore<=maxscore: #when others most likely don't give it such high score
                        print(f'{QUANT_ID} gave {quantifier_score} for the praise "{praise_reason}"')

    def _get_participant_metrics(self):
        # TODO: make this more modularized so the user can easily choose what analysis to include               
        quantifier_coef= []
        quantifier_score_displace=[]
        
        quantifier_rating_table=self.quantratingdf
        df = self.df
        for QUANT_ID in self.quantlist:
            quantdf = quantifier_rating_table.loc[quantifier_rating_table['QUANT_ID']==QUANT_ID]
            av_scores_others = []
            scoredist =[]
            scoredisplace=[]
            for kr, row in quantdf.iterrows():
                quantifier_score=row['QUANT_VALUE']
                praise_id = row['PRAISE_ID']
                praise_row=df.loc[df['ID']==praise_id]

                otherscores = praise_row.filter(like='SCORE ').values.tolist()[0]
                otherscores.remove(quantifier_score)
                otherscores=np.array(otherscores)
                av_scores_others.append(np.mean(otherscores))
                scoredist.append(np.sqrt(sum((quantifier_score - otherscores)**2))/(len(otherscores))) # like "standar deviation" from this score
                scoredisplace.append(np.mean(quantifier_score - otherscores))
            coef = np.corrcoef(quantdf['QUANT_VALUE'].values, av_scores_others)[1,0]
            quantifier_coef.append(coef)
            quantifier_score_displace.append(np.mean(scoredisplace))
        quantifier_metrics_df=pd.DataFrame(index=self.quantlist,data = {'pearson_coef':quantifier_coef,'av_score_displacement':quantifier_score_displace})        
        return quantifier_metrics_df
    
    def plot_coefficient(self):
        """returns a plotly express figure object"""
        quantifier_metrics_df = self.metrics_df
        fig=px.scatter(quantifier_metrics_df.sort_values(by='pearson_coef'),y='pearson_coef',title="How similiar is one's score with the other quantifiers?")
        return fig
    
    
    def plot_mean_displacement(self):
        """returns a plotly express figure object"""
        quantifier_metrics_df = self.metrics_df
        fig=px.bar(quantifier_metrics_df.sort_values(by='av_score_displacement'),y='av_score_displacement',title="Do one tends to give higher or lower scores than the other quantifiers?")
        return fig
        
        



