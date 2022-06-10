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


def load_configurations():
    exp_config = {}
    exp_config['description'] = "ann_ids_2017"
    exp_config['dataset_dir'] = "./Datasets/small_datasets/ids2017"
    exp_config['results_dir'] = "results/ann_ids17"
    exp_config['initial_split_ratio'] = 0.1

    classifier_config = {}
    classifier_config['input_nodes'] = 78
    classifier_config['output_nodes'] = 3
    classifier_config['ann_layer_units'] = [64]
    classifier_config['ann_layer_activations'] = ['relu']
    classifier_config['ann_layer_dropout_rates'] = [0.2]
    classifier_config['batch_size'] = 256
    classifier_config['epochs'] = 5
    classifier_config['early_stop_patience'] = 20
    classifier_config['tensorboard_log_dir'] = "results/ann_ids17"
    classifier_config['class_weights'] = 0

    al_config = {}
    al_config['query_strategy'] = 'uncertainty'
    al_config['selection_strategy'] = 'max_n'
    al_config['selection_param'] = 10

    ids_config = {}
    ids_config['al_batch_size'] = 100
    ids_config['classifier_confidence_th'] = 0.8
    ids_config['data_count_th'] = 50
    ids_config['al_selection_param'] = 10

    return exp_config, classifier_config, al_config, ids_config


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

    selected_classes = ['BENIGN', 'PortScan', 'DDoS']
    flt = y_train_i.isin(selected_classes)
    X_train_i_s = X_train_i.loc[flt]
    y_train_i_s = y_train_i.loc[flt]

    initial_datasets_orig = (X_train_i_s, y_train_i_s)
    remaining_datasets_orig = (X_train_r, y_train_r)

    return initial_datasets_orig, remaining_datasets_orig


def run_intrusion_detector(network_flow_data, true_labels, intrusion_detector):
    results = np.array([])
    for index, current_data in network_flow_data.iterrows():
        predicted_label = intrusion_detector.predict(current_data.values.reshape(1, 78), index)
        # true_label = true_labels[index]
        # print("predicted label = "+predicted_label+", true label = "+true_label)
        results = np.append(results, [predicted_label])
    return results


def evaluate_results(results, true_labels):
    pass


def run_experiment(exp_config, classifier_config, al_config, ids_config):
    # load dataset
    datasets_orig = load_datasets(exp_config['dataset_dir'])

    # prepare dataset
    initial_datasets_orig, remaining_datasets_orig = split_initial_data(datasets_orig, exp_config['initial_split_ratio'])
    X_train, y_train = remaining_datasets_orig
    X_train_c, y_train_c = initial_datasets_orig
    y_train_c_enc, label_encoder = utility.encode_data(y_train_c)

    # create classifier and pre-train them
    classifier = AttackClassifier(classifier_config)
    classifier.fit(X_train_c, y_train_c_enc)

    # create active learner
    labels = y_train
    active_learner = ActiveLearner(al_config['query_strategy'], al_config['selection_strategy'], al_config['selection_param'], labels)

    # create intrusion detector
    initial_train_data = X_train_c, y_train_c, label_encoder
    intrusion_detector = IntrusionDetector(active_learner, classifier, initial_train_data, ids_config)

    # run ids and improve ids with active learner
    results = run_intrusion_detector(X_train, y_train, intrusion_detector)
    evaluate_results(results, y_train)


def main():
    # Disable GPU as it appears to be slower than CPU (to enable GPU, comment out this line)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    exp_config, classifier_config, al_config, ids_config = load_configurations()
    run_experiment(exp_config, classifier_config, al_config, ids_config)


if __name__ == "__main__":
    main()
