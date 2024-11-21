import glob
import os
import pickle
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from utils import generate_waymo_simulation_submission


def load_pickle(pkl_file):
    with open(pkl_file, 'rb') as handle:
        prediction = pickle.load(handle)
    return prediction


def load_and_merge_pickles(directory):
    merged_prediction = {}

    pkl_files = glob.glob(os.path.join(directory, "*.pkl"))

    with ProcessPoolExecutor() as executor:
        futures = []

        # Submit each scenario to be processed in parallel
        for pkl_file in pkl_files:
            futures.append(executor.submit(load_pickle, pkl_file))

        # As the futures are completed, add the results to scenario_rollouts
        for future in tqdm(as_completed(futures), total=len(futures)):
            merged_prediction.update(future.result())

    return merged_prediction


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--submission_dir', type=str, required=True)
    args = parser.parse_args()

    pred = load_and_merge_pickles(args.root)

    generate_waymo_simulation_submission(
        pred,
        num_rollouts=32,
        account_name="Yueyang Wang",
        method_name="kGPT",
        authors=["Leon"],
        affiliation="",
        description="",
        method_link="",
        uses_lidar_data=False,
        uses_camera_data=False,
        uses_public_model_pretraining=False,
        public_model_names=["kGPT"],
        num_model_parameters="0.64M",
        acknowledge_complies_with_closed_loop_requirement=True,
        submission_dir=args.submission_dir,
        submission_file_name="submission",
    )
