import datetime
import pandas as pd


def generate_post(params, ROOT_INPUT_PATH=""):
    praise_path = params["system_settings"]["praise"]["input_files"]["praise_data"]
    token_table_path = ROOT_INPUT_PATH + \
        "distribution_results/raw_csv_exports/final_praise_token_allocation.csv"
    params["token_allocation_per_reward_system"] = list(
        map(float, params["token_allocation_per_reward_system"]))

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
