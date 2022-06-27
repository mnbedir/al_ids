from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, \
    recall_score, precision_score, f1_score

import utility
import sys, os, time, logging, csv, glob
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from active_learners.active_learner import ActiveLearner
from attack_classifier import AttackClassifier
from intrusion_detector_fast import IntrusionDetector

hdf_key = 'my_key'

input_and_pool_data_table = pd.DataFrame()
pool_and_selected_data_table = pd.DataFrame()
train_data_table = pd.DataFrame()
new_class_table = pd.DataFrame()
general_eval_info_table = pd.DataFrame()
detailed_eval_info_table = pd.DataFrame()
train_counter = 0


def load_configurations(config_file_path):
    txt_content = ""
    with open(config_file_path, 'r') as f:
        for line in f:
            txt_content += line

    config = eval(txt_content)
    return config['exp_config'], config['classifier_config'], config['al_config'], config['ids_config']


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
    accuracy_balanced = balanced_accuracy_score(y_test, y_test_pred)

    f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

    recall_weighted = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_test_pred, average='macro', zero_division=0)

    precision_weighted = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_test, y_test_pred, average='macro', zero_division=0)

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


def report_initial_info(train_data_info, evaluation_info):
    global train_data_table
    global new_class_table
    global general_eval_info_table
    global detailed_eval_info_table

    new_class_info = {'name': "train-0", 'new_class': 0}

    general_info, detailed_info = evaluation_info

    train_data_info_df = pd.DataFrame(train_data_info, index=[0])
    new_class_table_df = pd.DataFrame(new_class_info, index=[0])
    general_eval_info_df = pd.DataFrame(general_info, index=[0])
    detailed_eval_info_df = pd.DataFrame(detailed_info, index=[0])

    # table 3: What is new train data for each retain step?
    train_data_table = pd.concat([train_data_table, train_data_info_df], ignore_index=True)

    # table 4: Is there new class in train data?
    new_class_table = pd.concat([new_class_table, new_class_table_df], ignore_index=True)

    # table 5: What is the performance of retrained classifier on test data?
    general_eval_info_table = pd.concat([general_eval_info_table, general_eval_info_df], ignore_index=True)

    # table 6: What is the performance of retrained classifier on test data for each class?
    detailed_eval_info_table = pd.concat([detailed_eval_info_table, detailed_eval_info_df], ignore_index=True)


def report_info(data_info, evaluation_info):
    global input_and_pool_data_table
    global pool_and_selected_data_table
    global train_data_table
    global new_class_table
    global general_eval_info_table
    global detailed_eval_info_table

    input_data_info, data_pool_info, labeled_data_info, train_data_info, new_class_info = data_info
    general_info, detailed_info = evaluation_info

    input_data_info_df = pd.DataFrame(input_data_info, index=[0])
    data_pool_info_df = pd.DataFrame(data_pool_info, index=[0])
    labeled_data_info_df = pd.DataFrame(labeled_data_info, index=[0])
    train_data_info_df = pd.DataFrame(train_data_info, index=[0])
    new_class_table_df = pd.DataFrame(new_class_info, index=[0])

    general_eval_info_df = pd.DataFrame(general_info, index=[0])
    detailed_eval_info_df = pd.DataFrame(detailed_info, index=[0])

    logging.info("Evaluation on test data: \n" + str(general_eval_info_df))

    # table 1: Which classes confidence is low?
    input_and_pool_data_table = pd.concat([input_and_pool_data_table, input_data_info_df], ignore_index=True)
    input_and_pool_data_table = pd.concat([input_and_pool_data_table, data_pool_info_df], ignore_index=True)

    # table 2: Which classes selected by active learner?
    pool_and_selected_data_table = pd.concat([pool_and_selected_data_table, data_pool_info_df], ignore_index=True)
    pool_and_selected_data_table = pd.concat([pool_and_selected_data_table, labeled_data_info_df], ignore_index=True)

    # table 3: What is new train data for each retain step?
    train_data_table = pd.concat([train_data_table, train_data_info_df], ignore_index=True)

    # table 4: Is there new class in train data?
    new_class_table = pd.concat([new_class_table, new_class_table_df], ignore_index=True)

    # table 5: What is the performance of retrained classifier on test data?
    general_eval_info_table = pd.concat([general_eval_info_table, general_eval_info_df], ignore_index=True)

    # table 6: What is the performance of retrained classifier on test data for each class?
    detailed_eval_info_table = pd.concat([detailed_eval_info_table, detailed_eval_info_df], ignore_index=True)


def write_report(result_dir):
    global input_and_pool_data_table
    global pool_and_selected_data_table
    global train_data_table
    global new_class_table
    global general_eval_info_table
    global detailed_eval_info_table

    input_and_pool_data_table.set_index('name', inplace=True)
    pool_and_selected_data_table.set_index('name', inplace=True)
    train_data_table.set_index('name', inplace=True)
    new_class_table.set_index('name', inplace=True)
    general_eval_info_table.set_index('name', inplace=True)
    detailed_eval_info_table.set_index('name', inplace=True)

    input_and_pool_data_table.to_csv(result_dir + '/input_and_pool_data_table.csv', sep='\t')
    pool_and_selected_data_table.to_csv(result_dir + '/pool_and_selected_data_table.csv', sep='\t')
    train_data_table.to_csv(result_dir + '/train_data_table.csv', sep='\t')
    new_class_table.to_csv(result_dir + '/new_class_table.csv', sep='\t')
    general_eval_info_table.to_csv(result_dir + '/general_eval_info_table.csv', sep='\t')
    detailed_eval_info_table.to_csv(result_dir + '/detailed_eval_info_table.csv', sep='\t')


def plot_evaluation_graph(result_dir):
    global new_class_table
    global general_eval_info_table
    indices = general_eval_info_table.index.values.tolist()

    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(indices, general_eval_info_table['accuracy'], label='accuracy')
    plt.plot(indices, general_eval_info_table['f1_score_weighted'], label='f1_score_weighted')
    plt.plot(indices, general_eval_info_table['accuracy_balanced'], label='accuracy_balanced')
    plt.plot(indices, general_eval_info_table['f1_score_macro'], label='f1_score_macro')
    plt.scatter(indices, new_class_table['new_class'])
    plt.xticks(rotation=90)
    plt.title("Evaluation Results")
    plt.legend()
    plt.savefig(result_dir + '/evaluation_graph_all.png')

    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(indices, general_eval_info_table['accuracy'], label='accuracy')
    plt.plot(indices, general_eval_info_table['f1_score_weighted'], label='f1_score_weighted')
    plt.scatter(indices, new_class_table['new_class'])
    plt.xticks(rotation=90)
    plt.title("Evaluation Results")
    plt.legend()
    plt.savefig(result_dir + '/evaluation_graph_weighted.png')

    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(indices, general_eval_info_table['accuracy_balanced'], label='accuracy_balanced')
    plt.plot(indices, general_eval_info_table['f1_score_macro'], label='f1_score_macro')
    plt.scatter(indices, new_class_table['new_class'])
    plt.xticks(rotation=90)
    plt.title("Evaluation Results")
    plt.legend()
    plt.savefig(result_dir + '/evaluation_graph_unweighted.png')


def run_intrusion_detector(X_train, y_train, X_test, y_test, intrusion_detector):
    initial_train_data_info = intrusion_detector.extract_train_data_info()
    initial_evaluation_info = evaluate_intrusion_detector(intrusion_detector, X_test, y_test)
    report_initial_info(initial_train_data_info, initial_evaluation_info)

    data_chunk_size = 100
    current_data_count = 0
    remaining_data_count = X_train.shape[0]
    input_data = np.array(np.empty((0, 78)))
    data_ids = np.array([])

    logging.info("start_data_count: " + str(remaining_data_count))
    t0 = time.time()

    for index, current_data in X_train.iterrows():
        new_data = current_data.values.reshape(1, 78)
        input_data = np.append(input_data, new_data, axis=0)
        data_ids = np.append(data_ids, [index])

        current_data_count += 1
        remaining_data_count -= 1

        if current_data_count < data_chunk_size and remaining_data_count > 0:
            continue

        predicted_label, class_prob, data_info = intrusion_detector.predict(input_data, data_ids)

        current_data_count = 0
        input_data = np.array(np.empty((0, 78)))
        data_ids = np.array([])

        if data_info is not None:
            evaluation_info = evaluate_intrusion_detector(intrusion_detector, X_test, y_test)
            report_info(data_info, evaluation_info)

    exec_time = time.time() - t0
    logging.info("remaining_data_count: "+str(remaining_data_count))
    logging.info("Experiment eval time: "+str(exec_time))


def prepare_attack_dataset(datasets_orig):
    X_train_i, y_train_i = datasets_orig
    flt = y_train_i != 'BENIGN'
    X_train_i_attack_only = X_train_i.loc[flt]
    y_train_i_attack_only = y_train_i.loc[flt]
    return X_train_i_attack_only, y_train_i_attack_only


def extract_and_save_data_info(result_dir, datasets_orig):
    (X_train, y_train), (X_test, y_test) = datasets_orig

    classes, counts = np.unique(y_train, return_counts=True)
    train_info_table = get_info_table(classes, counts, "train data")
    train_df = pd.DataFrame(train_info_table, index=[0])
    train_df.set_index('name', inplace=True)

    classes, counts = np.unique(y_test, return_counts=True)
    test_info_table = get_info_table(classes, counts, "test data")
    test_df = pd.DataFrame(test_info_table, index=[0])
    test_df.set_index('name', inplace=True)

    dataset_info_table = pd.concat([train_df, test_df])
    dataset_info_table.to_csv(result_dir + '/dataset_info_table.csv', sep='\t')


def get_info_table(classes, counts, name):
    info = {'name': name}
    for i in range(classes.size):
        info[classes[i]] = counts[i]
    info['total'] = np.sum(counts)
    return info


def run_experiment(exp_config, classifier_config, al_config, ids_config):
    # load dataset
    datasets_orig = load_datasets(exp_config['dataset_dir'])
    (X_train, y_train), (X_test, y_test) = datasets_orig

    # extract_and_save_data_info(exp_config['results_dir'], datasets_orig)

    # prepare dataset
    initial_datasets_orig, remaining_datasets_orig = split_initial_data(datasets_orig,
                                                                        exp_config['initial_split_ratio'])
    X_rem_train, y_rem_train = remaining_datasets_orig
    X_pre_train, y_pre_train = initial_datasets_orig
    y_pre_train_enc, label_encoder = utility.encode_data(y_pre_train)

    # create classifier and pre-train them
    classifier = AttackClassifier(classifier_config)
    classifier.fit(X_pre_train, y_pre_train_enc)

    # create active learner
    labels = y_rem_train
    active_learner = ActiveLearner(al_config['query_strategy'], al_config['selection_strategy'],
                                   al_config['selection_param'], labels)

    # create intrusion detector
    initial_train_data = X_pre_train, y_pre_train, label_encoder
    intrusion_detector = IntrusionDetector(active_learner, classifier, initial_train_data, ids_config)

    # run ids and improve ids with active learner
    run_intrusion_detector(X_rem_train, y_rem_train, X_test, y_test, intrusion_detector)

    # write report to csv
    write_report(exp_config['results_dir'])
    plot_evaluation_graph(exp_config['results_dir'])


def main():
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

    config_file_path = sys.argv[1]
    exp_config, classifier_config, al_config, ids_config = load_configurations(config_file_path)

    create_result_dir(exp_config['results_dir'])
    shutil.copy(config_file_path, exp_config['results_dir'])

    run_experiment(exp_config, classifier_config, al_config, ids_config)


if __name__ == "__main__":
    main()
