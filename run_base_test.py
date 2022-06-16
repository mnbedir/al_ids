
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score
import utility

import sys, os, time, logging, csv, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from attack_classifier import AttackClassifier

hdf_key = 'my_key'


def load_configurations():
    # TODO: save config into file
    exp_config = {}
    exp_config['description'] = "ann_ids_2017"
    exp_config['dataset_dir'] = "./Datasets/small_datasets/ids2017"
    exp_config['results_dir'] = "results"

    classifier_config = {}
    classifier_config['input_nodes'] = 78
    classifier_config['output_nodes'] = 12
    classifier_config['ann_layer_units'] = [64]
    classifier_config['ann_layer_activations'] = ['relu']
    classifier_config['ann_layer_dropout_rates'] = [0.2]
    classifier_config['batch_size'] = 32
    classifier_config['epochs'] = 30
    classifier_config['early_stop_patience'] = 20
    classifier_config['tensorboard_log_dir'] = "results/ann_ids17"
    classifier_config['class_weights'] = 0

    return exp_config, classifier_config


def load_datasets(data_dir):
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


def evaluate_intrusion_detector(classifier, X_test, y_test, label_encoder):
    y_test_pred_enc = classifier.predict_classes(X_test)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_enc.flatten())

    accuracy = accuracy_score(y_test, y_test_pred)
    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    report.pop('accuracy')
    report.pop('macro avg')
    report.pop('weighted avg')

    general_info = { 'accuracy': accuracy, 'f1_score': f1_weighted}
    detailed_info = {}

    for class_name in report:
        class_report = report[class_name]
        detailed_info[class_name] = class_report['f1-score']

    evaluation_info = (general_info, detailed_info)
    return evaluation_info


def write_report(result_dir, evaluation_info):
    general_info, detailed_info = evaluation_info

    general_df = pd.DataFrame(general_info, index=[0])
    detailed_df = pd.DataFrame(detailed_info, index=[0])

    general_df.to_csv(result_dir + '/base_general_eval_info_table.csv', sep='\t')
    detailed_df.to_csv(result_dir + '/base_detailed_eval_info_table.csv', sep='\t')


def run_experiment(exp_config, classifier_config):
    # load dataset
    datasets_orig = load_datasets(exp_config['dataset_dir'])
    (X_train, y_train), (X_test, y_test) = datasets_orig
    y_train_enc, label_encoder = utility.encode_data(y_train)

    classifier = AttackClassifier(classifier_config)
    classifier.fit(X_train, y_train_enc)

    evaluation_info = evaluate_intrusion_detector(classifier, X_test, y_test, label_encoder)
    write_report(exp_config['results_dir'], evaluation_info)


def main():
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    exp_config, classifier_config = load_configurations()
    run_experiment(exp_config, classifier_config)


if __name__ == "__main__":
    main()