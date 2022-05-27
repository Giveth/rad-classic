import os
import subprocess
import json
import shutil
from pathlib import Path
from natsort import natsorted

import papermill as pm
import scrapbook as sb

from distribution_tools.praise import forum_post_generator
import argparse


parser = argparse.ArgumentParser(
    description='RAD main script')
parser.add_argument("-p", "--path", type=str, required=True,
                    help="Path to the folder in which we'll perform the analysis")

args = parser.parse_args()
input_parameters = args.path + "/parameters.json"


params = {}
with open(input_parameters, "r") as read_file:
    params = json.load(read_file)

# declare the paths where we want to save stuff as constants for easy reference
ROOT_INPUT_PATH = args.path


# quick conveniency check
ROOT_INPUT_PATH = ROOT_INPUT_PATH if ROOT_INPUT_PATH[-1] == "/" else (
    ROOT_INPUT_PATH+"/")

ROOT_OUTPUT_PATH = ROOT_INPUT_PATH + \
    params["results_output_folder"] + "/"

# RAW_DATA_OUTPUT_PATH = ROOT_OUTPUT_PATH + "data/"
NOTEBOOK_OUTPUT_PATH = ROOT_OUTPUT_PATH + "executed_notebooks/"
REPORT_OUTPUT_PATH = ROOT_OUTPUT_PATH + "reports/"
DISTRIBUTION_OUTPUT_PATH = ROOT_OUTPUT_PATH + "raw_csv_exports/"

# create output file structure:
Path(ROOT_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(NOTEBOOK_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(REPORT_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(DISTRIBUTION_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

# apply all specified reward systems
for i, reward_system in enumerate(params["employed_reward_systems"]):
	
    print(f"\n\
\t=========================================\n \t\t\t{reward_system.capitalize()}   \n\t=========================================")

    if params["token_allocation_per_reward_system"][i] == "0":
        print(f'>>> No reward allocation for {params["employed_reward_systems"][i]}, skipping.')
        continue

    print("\nDISTRIBUTION:")

    # prepare the parameter set we will send to the distribution and the folder with the notebook templates
    system_params = params["system_settings"][reward_system]
    system_params["total_tokens_allocated"] = params["token_allocation_per_reward_system"][i]
    system_params["distribution_name"] = params["distribution_name"]
    system_params["results_output_folder"] = params["results_output_folder"]

    DISTRIBUTION_NOTEBOOK_FOLDER = "./distribution_tools/" + reward_system + "/"

    # make sure the notebook finds the path to the files
    for file in system_params["input_files"]:
        system_params["input_files"][file] = os.path.abspath(
            os.path.join(ROOT_INPUT_PATH, system_params["input_files"][file]))

    # run all notebooks in the relevant distribution folder
    sorted_contents = natsorted(os.listdir(DISTRIBUTION_NOTEBOOK_FOLDER))
    for notebook in sorted_contents:
        # make sure we only use .ipynb files
        if not (notebook.endswith(".ipynb")):
            continue

        dist_input_path = DISTRIBUTION_NOTEBOOK_FOLDER + notebook
        dist_output_path = NOTEBOOK_OUTPUT_PATH + "output_" + notebook
        
        print(f"\n|---{notebook} :")

        pm.execute_notebook(
            dist_input_path,
            dist_output_path,
            parameters=system_params
        )

    # copy generated distribution files to results folder
    for output_csv in os.listdir():
        if not (output_csv.endswith(".csv")):
            continue

        csv_destination = DISTRIBUTION_OUTPUT_PATH + output_csv
        os.rename(output_csv, csv_destination)

    print("\nANALYSIS:")

    # prepare the parameter set we will use for analysis and the folder with the notebook templates
    analysis_params = {"dist_notebook_path": dist_output_path,
                       "input_files": system_params["input_files"],
                       "distribution_parameters": system_params,
                       }


    ANALYSIS_NOTEBOOK_FOLDER = "./analysis_tools/notebooks/" + reward_system + "/"

    if not os.path.isdir(ANALYSIS_NOTEBOOK_FOLDER):
        print(f'{reward_system} analysis notebook not provided, skip analysis.')
    else:
        # run all notebooks in the analysis folder
        sorted_contents = natsorted(os.listdir(ANALYSIS_NOTEBOOK_FOLDER))

        for notebook in sorted_contents:

            # make sure we only use .ipynb files
            if not (notebook.endswith(".ipynb")):
                continue

            nb_input_path = ANALYSIS_NOTEBOOK_FOLDER + notebook
            nb_destination_path = NOTEBOOK_OUTPUT_PATH + "output_" + notebook
            print(f"\n|---{notebook} :")

            pm.execute_notebook(
                nb_input_path,
                nb_destination_path,
                parameters=analysis_params
            )

            # copy generated csv files to results folder
            for output_csv in os.listdir():
                if not (output_csv.endswith(".csv")):
                    continue
                csv_destination = DISTRIBUTION_OUTPUT_PATH + output_csv
                os.rename(output_csv, csv_destination)

            # generate HTML report
            return_buf = subprocess.run(
                "jupyter nbconvert --log-level=0 --to html --TemplateExporter.exclude_input=True %s" % nb_destination_path, shell=True)

            # move it to right folder
            html_report_origin = nb_destination_path[:-6] + ".html"
            html_report_destination = REPORT_OUTPUT_PATH + params["distribution_name"] + "-" +\
                notebook[2:-6] + "-report.html"
            os.rename(html_report_origin, html_report_destination)

with open(DISTRIBUTION_OUTPUT_PATH + 'forum_post.md', 'w') as output:
    output.write(forum_post_generator.generate_post(params, ROOT_INPUT_PATH))
