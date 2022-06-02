import pandas as pd
import numpy as np
import plotly.express as px


class praise_round():
    def __init__(self, praisedata=pd.DataFrame(), quantifiertable=pd.DataFrame()):
        self.df = praisedata
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
                av_scores_others.append(np.mean(otherscores))
                # like "standar deviation" from this score
                scoredist.append(
                    np.sqrt(sum((quantifier_score - otherscores)**2))/(len(otherscores)))
                scoredisplace.append(np.mean(quantifier_score - otherscores))
            coef = np.corrcoef(
                quantdf['QUANT_VALUE'].values, av_scores_others)[1, 0]
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


def generate_post(params, ROOT_INPUT_PATH=""):
    praise_path = params["system_settings"]["praise"]["input_files"]["praise_data"]
    token_table_path = ROOT_INPUT_PATH + \
        "distribution_results/raw_csv_exports/final_praise_token_allocation.csv"
    params["token_allocation_per_reward_system"] = list(
        map(int, params["token_allocation_per_reward_system"]))

    data = pd.read_csv(praise_path)
    start_date = pd.to_datetime(data["DATE"].min()).date()

    end_date = pd.to_datetime(data["DATE"].max()).date()
    total_tokens = sum(params["token_allocation_per_reward_system"])
    praise_pct = int((
        int(params["token_allocation_per_reward_system"][0]) / total_tokens) * 100)
    sourcecred_pct = int((
        int(params["token_allocation_per_reward_system"][1]) / total_tokens) * 100)

    rewards_amt = int(params["system_settings"]["praise"]
                      ["token_reward_percentages"]["contributor_rewards"]*100)
    quant_amt = int(params["system_settings"]["praise"]
                    ["token_reward_percentages"]["quantifier_rewards"]*100)
    rewardboard_amt = int(params["system_settings"]["praise"]
                          ["token_reward_percentages"]["rewardboard_rewards"]*100)

    token_table = pd.read_csv(token_table_path)
    token_table = token_table[["USER IDENTITY", "TOTAL TO RECEIVE"]].copy()
    token_table.rename(
        columns={'USER IDENTITY': 'Username', 'TOTAL TO RECEIVE': 'Rewards in TEC'}, inplace=True)
    markdown_table = token_table.to_markdown(index=False)
    output = (f'''
# TEC Rewards Distribution - {params["distribution_name"]}  - {start_date.strftime("%d/%m/%y")} to {end_date.strftime("%d/%m/%y")}
This period covers praise given between **{start_date.strftime("%d %B %Y")} and {end_date.strftime("%d %B %Y")}**. We allocated **{total_tokens}** TEC tokens for rewards, with a **{praise_pct}:{sourcecred_pct}** split between Praise and Sourcecred. Some praise accounts still haven’t been activated so the total amount below will be less than what we set aside to distribute.

Out of the total rewards:

* {rewards_amt}% of the tokens were given as praise rewards :pray:
* {quant_amt}% distributed among quantifiers :balance_scale:
* {rewardboard_amt}% assigned to the reward board :memo:

This data has been reviewed by the Quantifiers and the Reward Board, and has been submitted for distribution to the [Reward Board DAO](https://xdai.aragon.blossom.software/#/rewardboardtec/)


You can check out the [full period analysis here](ADD LINK HERE). :bar_chart:

This post will be open to the community for review for 48 hours then submitted to the Reward Board for final execution. :heavy_check_mark:

The Rewards Distribution for this round is as follows:
''')

    output += markdown_table

    return output


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
