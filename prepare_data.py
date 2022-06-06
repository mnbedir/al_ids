import os, logging
from pprint import pformat
import utility
import pandas as pd
import numpy as np
import re, math


class Params:
    pass


# Script params
params = Params()

# Common params
params.hdf_key = 'my_key'
# ---- Small datasets
# params.output_dir = '../Datasets/small_datasets/ids2017'
# ---- Full datasets
# params.output_dir = '../Datasets/full_datasets/ids2017'
params.output_dir = 'Datasets/small_datasets/ids2017'

# IDS 2017 params
params.ids2017_small = True
params.ids2017_datasets_dir = 'Datasets/CIC_IDS_2017/MachineLearningCVE'
params.ids2017_files_list = [
                'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                'Monday-WorkingHours.pcap_ISCX.csv',
                'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',   # Issue with flows file
                'Tuesday-WorkingHours.pcap_ISCX.csv',
                'Wednesday-workingHours.pcap_ISCX.csv'
                ]

params.ids2017_hist_num_bins = 10000

params.ids2017_flows_dir = '../Datasets/CIC_IDS_2017/GeneratedLabelledFlows/TrafficLabelling'
params.ids2017_flow_seqs_max_flow_seq_length = 100
params.ids2017_flow_seqs_max_flow_duration_secs = 3


def initial_setup(output_dir, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info('Created directory: {}'.format(output_dir))

    # Setup logging
    log_filename = output_dir + '/' + 'run_log.log'

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, 'w+'),
                  logging.StreamHandler()],
        level=logging.INFO
    )
    logging.info('Initialized logging. log_filename = {}'.format(log_filename))

    logging.info('Running script with following parameters\n{}'.format(pformat(params.__dict__)))


def print_dataset_sizes(datasets):
    (X_train, y_train), (X_test, y_test) = datasets
    logging.info("No. of features = {}".format(X_train.shape[1]))
    logging.info("Training examples = {}".format(X_train.shape[0]))
    logging.info("Test examples = {}".format(X_test.shape[0]))


def prepare_ids2017_datasets(params):
    # Load data_prep
    logging.info('Loading datasets')
    data_files_list = [params.ids2017_datasets_dir + '/' + filename for filename in params.ids2017_files_list]
    all_data = utility.load_datasets(data_files_list, header_row=0, strip_col_name_spaces=True)
    # utility.print_info(all_data)

    # Remove unicode values in class labels
    logging.info('Converting unicode labels to ascii')
    all_data['Label'] = all_data['Label'].apply(lambda x: x.encode('ascii', 'ignore').decode("utf-8"))
    all_data['Label'] = all_data['Label'].apply(lambda x: re.sub(' +', ' ', x)) # Remove double spaces

    # Following type conversion and casting (both) are necessary to convert the values in cols 14, 15 detected as objects
    # Otherwise, the training algorithm does not work as expected
    logging.info('Converting object type in columns 14, 15 to float64')
    all_data['Flow Bytes/s'] = all_data['Flow Bytes/s'].apply(lambda x: np.float64(x))
    all_data['Flow Packets/s'] = all_data['Flow Packets/s'].apply(lambda x: np.float64(x))
    all_data['Flow Bytes/s'] = all_data['Flow Bytes/s'].astype(np.float64)
    all_data['Flow Packets/s'] = all_data['Flow Packets/s'].astype(np.float64)

    # Remove some invalid values/ rows in the dataset
    # nan_counts = all_data.isna().sum()
    # logging.info(nan_counts)
    logging.info('Removing invalid values (inf, nan)')
    prev_rows = all_data.shape[0]
    all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_data.dropna(inplace=True)  # Some rows (1358) have NaN values in the Flow Bytes/s column. Get rid of them
    logging.info('Removed no. of rows = {}'.format(prev_rows - all_data.shape[0]))

    # Remove samples from classes with a very small no. of samples (cannot split with those classes)
    logging.info('Removing instances of rare classes')
    rare_classes = ['Infiltration', 'Web Attack Sql Injection', 'Heartbleed']
    all_data.drop(all_data[all_data['Label'].isin(rare_classes)].index, inplace=True)  # Inplace drop

    # Check class labels
    label_counts, label_perc = utility.count_labels(all_data['Label'])
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))

    X = all_data.loc[:, all_data.columns != 'Label']  # All columns except the last
    y = all_data['Label']

    # Take only 1% as the small subset
    if params.ids2017_small:
        logging.info('Splitting datset into 2 (small subset, discarded)')
        splits = utility.split_dataset(X, y, [0.1, 0.9])
        (X, y), (discarded, discarded) = splits
        logging.info('Small subset no. of examples = {}'.format(X.shape[0]))

    # Split into 2 sets (train, test)
    logging.info('Splitting training set into 3 (train, validation, test)')
    splits = utility.split_dataset(X, y, [0.8, 0.2])
    (X_train, y_train), (X_test, y_test) = splits

    # Save data_prep files in HDF format
    logging.info('Saving prepared datasets (train, test) to: {}'.format(params.output_dir))

    utility.write_to_hdf(X_train, params.output_dir + '/' + 'X_train.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_train, params.output_dir + '/' + 'y_train.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_test, params.output_dir + '/' + 'X_test.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_test, params.output_dir + '/' + 'y_test.h5', params.hdf_key, 5)

    logging.info('Saving complete')

    print_dataset_sizes(splits)


def add_additional_items_to_dict(dict, extra_char):
    new_dict = {}
    for key, val in dict.items():
        new_key = key + extra_char
        new_dict[new_key] = val

    dict.update(new_dict)


def extract_scale_and_write(X_df, y_df, indexes_to_extract, scaler_ob, suffix_str, suffic_index):
    X_extracted = X_df.loc[indexes_to_extract, X_df.columns != 'Label']
    y_extracted = y_df.loc[indexes_to_extract]

    columns = list(range(0, X_extracted.shape[1]))
    X_scaled = utility.scale_dataset(X_extracted, scaler=scaler_ob, columns=columns)

    X_filename = params.output_dir + '/' + 'X_' + suffix_str + '_' + str(suffic_index) + '.h5'
    y_filename = params.output_dir + '/' + 'y_' + suffix_str + '_' + str(suffic_index) + '.h5'

    utility.write_to_hdf(X_scaled, X_filename, params.hdf_key, 5, format='table')
    utility.write_to_hdf(y_extracted, y_filename, params.hdf_key, 5, format='table')


def shrink_dataset(X_info_df, y_df, shrink_to_rate):
    jump = math.ceil(1/shrink_to_rate)

    assert X_info_df.shape[0] == y_df.shape[0]

    X_info_shrunk = X_info_df.iloc[::jump, :]
    y_shrunk = y_df.iloc[::jump, :]

    logging.info('No. of rows after shrinking dataset: {}'.format(X_info_shrunk.shape[0]))


def main():
    initial_setup(params.output_dir, params)
    prepare_ids2017_datasets(params)  # Small subset vs. full is controlled by config flag
    logging.info('Data preparation complete')


if __name__ == "__main__":
    main()

