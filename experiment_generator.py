import json
import os
import logging


def load_base_config(config_file_path):
    txt_content = ""
    with open(config_file_path, 'r') as f:
        for line in f:
            txt_content += line

    config = eval(txt_content)
    return config


def create_experiments_dir(config_dir_path):
    os.makedirs(config_dir_path, exist_ok=True)
    logging.info('Created config directory: {}'.format(config_dir_path))


def write_to_file(config_dir_path, filename, config):
    path = config_dir_path + "/" + filename
    with open(path, 'w') as f:
        f.write(str(config))


def generate_experiments(config, config_name, param_name, val_list):
    exp_group_name = "exp_entropy_" + param_name

    config_dir_path = "exp_configs_entropy/" + exp_group_name
    create_experiments_dir(config_dir_path)

    for val in val_list:
        exp_name = exp_group_name + "_" + str(val)

        config['exp_config']['results_dir'] = "results/" + exp_group_name + "/" + exp_name
        config[config_name][param_name] = val

        filename = exp_name + "_config.txt"
        write_to_file(config_dir_path, filename, config)


def main():
    config_file_path = "default_config.txt"
    # config = load_base_config(config_file_path)

    # exp_config: initial_split_ratio experiments
    config = load_base_config(config_file_path)
    config_name = 'exp_config'
    param_name = 'initial_split_ratio'
    val_list = [0.01, 0.01, 0.03, 0.05]
    generate_experiments(config, config_name, param_name, val_list)

    # classifier_config: batch_size experiments
    config = load_base_config(config_file_path)
    config_name = 'classifier_config'
    param_name = 'batch_size'
    val_list = [128, 256, 512]
    generate_experiments(config, config_name, param_name, val_list)

    # classifier_config: epochs experiments
    config = load_base_config(config_file_path)
    config_name = 'classifier_config'
    param_name = 'epochs'
    val_list = [10, 20, 30]
    generate_experiments(config, config_name, param_name, val_list)

    # al_config: selection_param experiments
    config = load_base_config(config_file_path)
    config_name = 'al_config'
    param_name = 'selection_param'
    val_list = [50, 100, 250, 500]
    generate_experiments(config, config_name, param_name, val_list)

    # ids_config: pool_size_th experiments
    config = load_base_config(config_file_path)
    config_name = 'ids_config'
    param_name = 'pool_size_th'
    val_list = [500, 750, 1000]
    generate_experiments(config, config_name, param_name, val_list)

    # ids_config: new_class_min_count_th experiments
    config = load_base_config(config_file_path)
    config_name = 'ids_config'
    param_name = 'new_class_min_count_th'
    val_list = [1, 5, 10]
    generate_experiments(config, config_name, param_name, val_list)

    # ids_config: classifier_confidence_th experiments
    # ids_config: retrain_only_new_data experiments


if __name__ == "__main__":
    main()

