from fontTools.misc.classifyTools import Classifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score

import utility
import models.ann

import sys, os, time, logging, csv, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib

# Common params
from active_learners.active_learner import ActiveLearner
from attack_classifier import AttackClassifier
from intrusion_detector import IntrusionDetector

hdf_key = 'my_key'
data_info_table = pd.DataFrame()
evaluation_info_table = pd.DataFrame()


def setup_logging(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info('Created directory: {}'.format(output_dir))

    # Setup logging
    log_filename = output_dir + '/' + 'run_log.log'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, 'w+'),
                  logging.StreamHandler()],
        level=logging.INFO
    )
    logging.info('Initialized logging. log_filename = {}'.format(log_filename))


def load_configurations():
    # TODO: save config into file
    exp_config = {}
    exp_config['description'] = "ann_ids_2017"
    exp_config['dataset_dir'] = "./Datasets/small_datasets/ids2017"
    exp_config['results_dir'] = "results"
    exp_config['initial_split_ratio'] = 0.05

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
    al_config['selection_param'] = 50

    ids_config = {}
    ids_config['al_batch_size'] = 500
    ids_config['classifier_confidence_th'] = 0.8
    ids_config['data_count_th'] = 10

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


def evaluate_intrusion_detector(intrusion_detector, X_test, y_test):
    classifier = intrusion_detector.get_attack_classifier()
    label_encoder = intrusion_detector.get_label_encoder()

    y_test_pred_enc = classifier.predict_classes(X_test)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_enc.flatten())

    accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    info = {'accuracy': accuracy, 'f1_score': f1}
    return info


def report_info(data_info, evaluation_info):
    global data_info_table
    global evaluation_info_table

    data_pool_info, labeled_data_info, train_data_info = data_info
    data_pool_info_df = pd.DataFrame(data_pool_info, index=[0])
    labeled_data_info_df = pd.DataFrame(labeled_data_info, index=[0])
    train_data_info_df = pd.DataFrame(train_data_info, index=[0])

    data_info_table = pd.concat([data_info_table, data_pool_info_df], ignore_index=True)
    data_info_table = pd.concat([data_info_table, labeled_data_info_df], ignore_index=True)
    data_info_table = pd.concat([data_info_table, train_data_info_df], ignore_index=True)

    evaluation_info_df = pd.DataFrame(evaluation_info, index=[0])
    evaluation_info_table = pd.concat([evaluation_info_table, evaluation_info_df], ignore_index=True)


def write_report(result_dir):
    global data_info_table
    global evaluation_info_table

    data_info_table.to_csv(result_dir + '/data_info_table.csv', sep='\t')
    evaluation_info_table.to_csv(result_dir + '/evaluation_info_table.csv', sep='\t')


def plot_evaluation_graph(result_dir):
    global evaluation_info_table
    indices = evaluation_info_table.index.values.tolist()

    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(indices, evaluation_info_table['f1_score'], label='f1_score')
    plt.plot(indices, evaluation_info_table['accuracy'], label='accuracy')
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title("Simple Plot")
    plt.legend()

    plt.savefig(result_dir+'/evaluation_graph.png')
    plt.show()


def run_intrusion_detector(X_train, y_train, X_test, y_test, intrusion_detector):
    results = np.array([])
    counter = 0

    # initial_evaluation_info = evaluate_intrusion_detector(intrusion_detector, X_test, y_test)
    # initial_data_info = None
    # report_info(initial_data_info, initial_evaluation_info)

    for index, current_data in X_train.iterrows():
        predicted_label, class_prob, data_info = intrusion_detector.predict(current_data.values.reshape(1, 78), index)
        counter += 1
        # true_label = y_train[index]
        # if predicted_label == true_label:
        #     logging.info(str(counter) + " predicted label = " + predicted_label + "(" + str(class_prob) + ")")
        # else:
        #     logging.info(str(counter) + " predicted label = " + predicted_label + "(" + str(class_prob) + "), but true label = " + true_label)

        if data_info is not None:
            evaluation_info = evaluate_intrusion_detector(intrusion_detector, X_test, y_test)
            report_info(data_info, evaluation_info)

        if counter == 5005:
            break


def run_experiment(exp_config, classifier_config, al_config, ids_config):
    # load dataset
    datasets_orig = load_datasets(exp_config['dataset_dir'])
    (X_train, y_train), (X_test, y_test) = datasets_orig

    # prepare dataset
    initial_datasets_orig, remaining_datasets_orig = split_initial_data(datasets_orig, exp_config['initial_split_ratio'])
    X_train, y_train = remaining_datasets_orig
    X_pre_train, y_pre_train = initial_datasets_orig
    y_pre_train_enc, label_encoder = utility.encode_data(y_pre_train)

    # create classifier and pre-train them
    classifier = AttackClassifier(classifier_config)
    classifier.fit(X_pre_train, y_pre_train_enc)

    # create active learner
    labels = y_train
    active_learner = ActiveLearner(al_config['query_strategy'], al_config['selection_strategy'], al_config['selection_param'], labels)

    # create intrusion detector
    initial_train_data = X_pre_train, y_pre_train, label_encoder
    intrusion_detector = IntrusionDetector(active_learner, classifier, initial_train_data, ids_config)

    # run ids and improve ids with active learner
    run_intrusion_detector(X_train, y_train, X_test, y_test, intrusion_detector)

    # write report to csv
    write_report(exp_config['results_dir'])
    plot_evaluation_graph(exp_config['results_dir'])


def main():
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    exp_config, classifier_config, al_config, ids_config = load_configurations()
    run_experiment(exp_config, classifier_config, al_config, ids_config)


if __name__ == "__main__":
    main()
