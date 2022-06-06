from fontTools.misc.classifyTools import Classifier

import utility
import models.ann

import sys, os, time, logging, csv, glob
import pandas as pd
import numpy as np

import matplotlib

# Common params
from active_learners.active_learner import ActiveLearner
from anomaly_detector import AnomalyDetector
from attack_classifier import AttackClassifier
from intrusion_detector import IntrusionDetector

hdf_key = 'my_key'


def load_params():
    exp_params = {}
    exp_params['description'] = "ann_ids_2017"
    exp_params['dataset_dir'] = "./Datasets/small_datasets/ids2017"
    exp_params['results_dir'] = "results/ann_ids17"
    exp_params['initial_split_ratio'] = 0.1

    exp_params['id_batch_size'] = 3
    exp_params['confidence_th'] = 0.8

    # Classifier configuration
    exp_params['input_nodes'] = 78
    exp_params['output_nodes'] = 3
    exp_params['ann_layer_units'] = [64]
    exp_params['ann_layer_activations'] = ['relu']
    exp_params['ann_layer_dropout_rates'] = [0.2]
    exp_params['epochs'] = 3
    exp_params['batch_size'] = 50
    exp_params['early_stop_patience'] = 20
    exp_params['tensorboard_log_dir'] = "results/ann_ids17"
    exp_params['class_weights'] = 0

    return exp_params


def load_datasets(data_dir):
    # Lists of train, val, test files (X and y)
    X_train_files = glob.glob(data_dir + '/' + 'X_train*')
    y_train_files = glob.glob(data_dir + '/' + 'y_train*')

    X_test_files = glob.glob(data_dir + '/' + 'X_test*')
    y_test_files = glob.glob(data_dir + '/' + 'y_test*')

    X_train_files.sort()
    y_train_files.sort()

    X_test_files.sort()
    y_test_files.sort()

    assert len(X_train_files) > 0
    assert len(y_train_files) > 0

    X_train_dfs = [utility.read_hdf(file, hdf_key) for file in X_train_files]
    X_test_dfs = [utility.read_hdf(file, hdf_key) for file in X_test_files]
    y_train_dfs = [utility.read_hdf(file, hdf_key) for file in y_train_files]
    y_test_dfs = [utility.read_hdf(file, hdf_key) for file in y_test_files]

    X_train = pd.concat(X_train_dfs)
    X_test = pd.concat(X_test_dfs)
    y_train = pd.concat(y_train_dfs)
    y_test = pd.concat(y_test_dfs)

    return (X_train, y_train), (X_test, y_test)


def split_initial_data(datasets_orig, split_ratio):
    (X_train, y_train), (X_test, y_test) = datasets_orig

    total_element_count = X_train.shape[0]
    initial_element_size = int(split_ratio * total_element_count)
    remaining_element_size = total_element_count - initial_element_size

    X_train_i = X_train.head(initial_element_size)
    y_train_i = y_train.head(initial_element_size)

    X_train_r = X_train.tail(remaining_element_size)
    y_train_r = y_train.tail(remaining_element_size)

    selected_classes = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS']
    flt = y_train_i.isin(selected_classes)
    X_train_i_s = X_train_i.loc[flt]
    y_train_i_s = y_train_i.loc[flt]

    initial_datasets_orig = (X_train_i_s, y_train_i_s)
    remaining_datasets_orig = (X_train_r, y_train_r)

    return initial_datasets_orig, remaining_datasets_orig


def convert_to_binary(y):
    if y != 'BENIGN':
        y = 'ATTACK'
    return y


def prepare_anomaly_detector_dataset(initial_datasets_orig):
    X_train_i, y_train_i = initial_datasets_orig
    # convert labels to binary
    y_train_i_binary = y_train_i.apply(convert_to_binary)
    return X_train_i, y_train_i_binary


def prepare_classifier_dataset(initial_datasets_orig):
    X_train_i, y_train_i = initial_datasets_orig
    flt = y_train_i != 'BENIGN'
    X_train_i_attack_only = X_train_i.loc[flt]
    y_train_i_attack_only = y_train_i.loc[flt]
    return X_train_i_attack_only, y_train_i_attack_only


def encode_data(y_train):
    # One-hot encode class labels (needed as output layer has multiple nodes)
    label_encoder, unused = utility.encode_labels(y_train, encoder=None)
    unused, y_train_enc = utility.encode_labels(y_train, encoder=label_encoder)
    return y_train_enc, label_encoder


def run_intrusion_detector(network_flow_data, intrusion_detector):
    results = np.array([])
    for index, current_data in network_flow_data.iterrows():
        predicted_label = intrusion_detector.predict(current_data.values.reshape(1, 78), index)
        results = np.append(results, [predicted_label])
    return results


def evaluate_results(results, true_labels):
    pass


def run_anomaly_detector(network_flow_data, anomaly_detector):
    results = np.array([])
    for index, current_data in network_flow_data.iterrows():
        predicted_label = anomaly_detector.predict(current_data.values.reshape(1, 78))
        np.append(results, [predicted_label])
    return results


def run_experiment(exp_params):
    # load dataset
    datasets_orig = load_datasets(exp_params['dataset_dir'])

    # prepare dataset
    initial_datasets_orig, remaining_datasets_orig = split_initial_data(datasets_orig,
                                                                        exp_params['initial_split_ratio'])
    X_train, y_train = remaining_datasets_orig
    X_train_ad, y_train_ad = prepare_anomaly_detector_dataset(initial_datasets_orig)
    X_train_c, y_train_c = prepare_classifier_dataset(initial_datasets_orig)
    y_train_c_enc, label_encoder = encode_data(y_train_c)

    # create anomaly detector and classifier and pre-train them
    anomaly_detector = AnomalyDetector()
    anomaly_detector.fit(X_train_ad.to_numpy(), y_train_ad.to_numpy())

    # results = run_anomaly_detector(X_train, anomaly_detector)

    classifier = AttackClassifier(exp_params)
    classifier.fit(X_train_c, y_train_c_enc)

    # create active learner
    query_strategy = 'uncertainty'
    selection_strategy = 'max_n'
    selection_param = 1
    labels = y_train
    active_learner = ActiveLearner(query_strategy, selection_strategy, selection_param, labels)

    # create intrusion detector
    initial_train_data = X_train_c, y_train_c_enc, label_encoder

    intrusion_detector = IntrusionDetector(anomaly_detector, classifier, active_learner, initial_train_data,
                                           exp_params['id_batch_size'], exp_params['confidence_th'])

    # run ids and improve ids with active learner
    results = run_intrusion_detector(X_train, intrusion_detector)
    evaluate_results(results, y_train)


def main():
    # Disable GPU as it appears to be slower than CPU (to enable GPU, comment out this line)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    exp_params = load_params()
    run_experiment(exp_params)


if __name__ == "__main__":
    main()
