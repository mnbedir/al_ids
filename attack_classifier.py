from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
import logging, hashlib
from keras_callbacks import ModelCheckpointer, Metrics, WeightRestorer, StopperOnGoal
import utility
from sklearn.utils import class_weight
import numpy as np


class AttackClassifier:

    def __init__(self, exp_params):
        self.params = exp_params
        self.input_nodes = exp_params['input_nodes']
        self.output_nodes = exp_params['output_nodes']
        self.layer_nodes = exp_params['ann_layer_units']
        self.activations = exp_params['ann_layer_activations']
        self.dropouts = exp_params['ann_layer_dropout_rates']

        self.params['batch_size'] = exp_params['batch_size']
        self.params['epochs'] = exp_params['epochs']
        self.params['early_stop_patience'] = exp_params['early_stop_patience']

        self.ann = None
        self.create_ann()

    def reinit(self, new_output_node_count):
        self.output_nodes = new_output_node_count
        # self.create_ann()

        self.ann.pop()
        self.ann.add(Dense(self.output_nodes, activation='softmax',
                           kernel_initializer='glorot_uniform', bias_initializer='zeros'))

        self.ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def create_ann(self):
        self.ann = Sequential()
        # First hidden layer (need to specify input layer nodes here)
        self.ann.add(Dense(self.layer_nodes[0], input_dim=self.input_nodes, activation=self.activations[0],
                           kernel_initializer='he_uniform', bias_initializer='zeros'))
        self.ann.add(BatchNormalization())
        self.ann.add(Dropout(self.dropouts[0]))

        # Other hidden layers
        for i in range(1, len(self.layer_nodes)):
            self.ann.add(Dense(self.layer_nodes[i], activation=self.activations[i],
                               kernel_initializer='he_uniform', bias_initializer='zeros'))
            self.ann.add(BatchNormalization())  # BN after Activation (contested, but current preference)
            self.ann.add(Dropout(self.dropouts[i]))

        # Output layer
        self.ann.add(Dense(self.output_nodes, activation='softmax',
                           kernel_initializer='glorot_uniform', bias_initializer='zeros'))

        self.ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        #logger = logging.getLogger(__name__)
        #self.ann.summary(print_fn=logger.info)

    def set_layers_weights(self, all_layers_weights):
        dense_layers = [layer for layer in self.ann.layers if type(layer) is Dense]
        dense_layers = dense_layers[:-1]  # Ignore the output layer

        assert len(dense_layers) == len(all_layers_weights)

        for layer, weights in zip(dense_layers, all_layers_weights):
            layer.set_weights(weights)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        epochs = self.params['epochs']
        # early_stop_patience = self.params['early_stop_patience']
        batch_size = self.params['batch_size']

        # --------------------------------
        # Split dataset (for special experiments only)

        # if 'dataset_split_ratio' in self.params and self.params['dataset_split_ratio'] > 0:
        #     split_ratio = self.params['dataset_split_ratio']
        #
        #     random_seed = None
        #     if 'split_random_seed' in self.params:
        #         random_seed = self.params['split_random_seed']
        #
        #     logging.info('Splitting train set into 2 sets (set1, set2), random_seed = {}'.format(random_seed))
        #     splits = utility.split_dataset(X_train, y_train, [split_ratio, 1 - split_ratio], random_seed)
        #     (X_train_set1, y_train_set1), (X_train_set2, y_train_set2) = splits
        #     dataset_hash = hashlib.sha1(str(X_train_set1).encode('utf-8')).hexdigest()
        #     logging.info('Split sizes (instances). total = {}, set1 = {}, set2 = {}, set1 dataset hash = {}'
        #                  .format(X_train.shape[0], X_train_set1.shape[0], X_train_set2.shape[0], dataset_hash))
        #
        #     X_train, y_train = (X_train_set2, y_train_set2)  # Train on set2

        # --------------------------------
        # Create callbacks

        # tensorboard_cb = TensorBoard(log_dir=self.params['tensorboard_log_dir'],)
        #
        # interval = max(epochs // 10, 10)
        # checkpointer_cb = ModelCheckpointer(model_wrapper=self, checkpoint_epoch_interval=interval,
        #                                     save_directory=self.params['results_dir'], filename_prefix='ann')
        #
        # callbacks = [tensorboard_cb, checkpointer_cb]
        # # callbacks = [tensorboard_cb, checkpointer_cb, metrics_cb]
        #
        # # Stop training when a goal metric value is reached
        # # if 'goal_metric' in self.params and self.params['goal_metric'] > 0:
        # #     assert X_test is not None
        # #     assert y_test is not None
        # #     if parent is None:
        # #         model_wrapper = self
        # #     else:
        # #         model_wrapper = parent
        # #     goal_stopper_cb = StopperOnGoal(model_wrapper, X_test, y_test, self.params['goal_metric'], "accuracy")
        # #     callbacks.append(goal_stopper_cb)
        #
        # if X_valid is not None:
        #     weight_restorer_cb = WeightRestorer(epochs)
        #     callbacks.append(weight_restorer_cb)
        #
        # # Early stopping is enabled
        # if X_valid is not None and early_stop_patience > 0 and early_stop_patience < epochs:
        #     early_stopping_cb = EarlyStopping(monitor='val_loss', patience=early_stop_patience,
        #                                       verbose=1, mode='auto', restore_best_weights=False)
        #     callbacks.append(early_stopping_cb)

        # --------------------------------
        # Class weights to according to imbalance

        class_weights = None
        d_class_weights = None
        if self.params['class_weights'] != 0:
            logging.info("Computing class weights to according to imbalance")
        y_ints = [y.argmax() for y in y_train]
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_ints), y=y_ints)
        d_class_weights = dict(enumerate(class_weights))

        #     logging.info("Class weights below.\n{}".format(class_weights))

        # --------------------------------
        # Fit

        if X_valid is not None and y_valid is not None:
            validation_data = (X_valid, y_valid)
        else:
            validation_data = None

        # metrics_cb = Metrics(val_data=validation_data)
        #
        # callbacks.append(metrics_cb)
        # callbacks = callbacks,
        history = self.ann.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                               validation_data=validation_data,
                               class_weight=d_class_weights, verbose=1)

        # history.history['val_f1'] = metrics_cb.val_f1s

        # --------------------------------
        # Evaluate loss of final (best-weights-restored) models
        # These values can be different from the values seen during training (due to batch norm and dropout)

        train_loss = self.ann.evaluate(X_train, y_train, verbose=0)
        # val_loss = -1
        # if X_valid is not None and y_valid is not None:
        #     val_loss = self.ann.evaluate(X_valid, y_valid, verbose=0)
        # logging.info('Last epoch loss evaluation: train_loss = {:.6f}, val_loss = {:.6f}'.format(train_loss, val_loss))

        return history

    def predict(self, X):
        # return self.predict_classes(X)
        return self.ann.predict(X, verbose=0)

    def predict_classes(self, X):
        predict_x = self.ann.predict(X, verbose=0)
        classes_x = np.argmax(predict_x, axis=1)
        return classes_x

    def get_last_hidden_layer_activations(self, X):
        # Temp models from input to last layer
        last_layer_model = Model(inputs=self.ann.input,
                                 outputs=self.ann.layers[-2].output)  # -2 is the last hidden layer

        last_layer_output = last_layer_model.predict(X)
        return last_layer_output

    def save(self, filename, **kwargs):
        self.ann.save(filename, kwargs)
