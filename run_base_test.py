import shutil

from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, balanced_accuracy_score, \
    recall_score
import utility

import sys, os, time, logging, csv, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from attack_classifier import AttackClassifier

hdf_key = 'my_key'


def load_configurations(config_file_path):
    txt_content = ""
    with open(config_file_path, 'r') as f:
        for line in f:
            txt_content += line

    config = eval(txt_content)
    return config['exp_config'], config['classifier_config']


def create_result_dir(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    logging.info('Created result directory: {}'.format(results_dir))


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


def evaluate_classifier(intrusion_detector, X_test, y_test):
    classifier = intrusion_detector.get_attack_classifier()
    label_encoder = intrusion_detector.get_label_encoder()

    y_test_pred_enc = classifier.predict_classes(X_test)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_enc.flatten())

    accuracy = accuracy_score(y_test, y_test_pred)
    accuracy_balanced = balanced_accuracy_score(y_test, y_test_pred)

    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    f1_macro = f1_score(y_test, y_test_pred, average='macro')

    recall_weighted = recall_score(y_test, y_test_pred, average='weighted')
    recall_macro = recall_score(y_test, y_test_pred, average='macro')

    precision_weighted = precision_score(y_test, y_test_pred, average='weighted')
    precision_macro = precision_score(y_test, y_test_pred, average='macro')

    report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    report.pop('accuracy')
    report.pop('macro avg')
    report.pop('weighted avg')

    global train_counter
    name = "train-" + str(train_counter)
    general_info = {'name': name,

                    'accuracy': accuracy,
                    'accuracy_balanced': accuracy_balanced,

                    'f1_score_weighted': f1_weighted,
                    'f1_score_macro': f1_macro,

                    'precision_weighted': precision_weighted,
                    'precision_macro': precision_macro,

                    'recall_weighted': recall_weighted,
                    'recall_macro': recall_macro
                    }

    detailed_info = {'name': name}

    for class_name in report:
        class_report = report[class_name]
        detailed_info[class_name] = class_report['f1-score']

    evaluation_info = (general_info, detailed_info)
    train_counter += 1
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

    evaluation_info = evaluate_classifier(classifier, X_test, y_test, label_encoder)
    write_report(exp_config['results_dir'], evaluation_info)


def main():
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

    # config_file_path = sys.argv[1]
    config_file_path = "default_base_config.txt"
    exp_config, classifier_config = load_configurations(config_file_path)

    create_result_dir(exp_config['results_dir'])
    shutil.copy(config_file_path, exp_config['results_dir'])

    run_experiment(exp_config, classifier_config)


if __name__ == "__main__":
    main()
