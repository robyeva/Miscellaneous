__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""Freely inspired by the LinkedIn Learning course: 
Building Deep Learning Applications with Keras 2.0, by Adam Geitgey """

import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf

path = '../Exercise Files/'


def scale_data(filename_train, filename_test, folder='03/'):
    """Scale data to the [0,1] range

    :param filename_train: str
              Filename containing train data (.csv)

    :param filename_test: str
              Filename containing test data (.csv)

    :param folder: str
                Folder where data are stored (Exercise Files)

    :return:

        scaled_train_filename: str
                    Filename of scaled train data
        scaled_test_filename: str
                    Filename of scaled test data
        scale_factor: float
                    Divisive scaling factor to be used to scale data back
        subtract_factor: float
                    Subtractive scaling factor to be used to scale data back
    """

    # Load data
    training_data_df = pd.read_csv(path + folder + filename_train)
    test_data_df = pd.read_csv(path + folder + filename_test)

    # Data needs to be scaled to a small range [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_training = scaler.fit_transform(training_data_df)
    # Use scaler.transform to apply the same scaling as the train dataset
    scaled_testing = scaler.transform(test_data_df)

    # Print out the adjustment that the scaler applied to the total_earnings column of data (needed to scale back
    # predictions)
    scale_factor = scaler.scale_[8]
    subtract_factor = scaler.min_[8]
    print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scale_factor,
                                                                                                       subtract_factor))

    scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
    scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

    # Save scaled data dataframes to new CSV files
    scaled_train_filename = "sales_data_training_scaled.csv"
    scaled_test_filename = "sales_data_test_scaled.csv"

    scaled_training_df.to_csv(path + scaled_train_filename, index=False)
    scaled_testing_df.to_csv(path + scaled_test_filename, index=False)

    return scaled_train_filename, scaled_test_filename, scale_factor, subtract_factor


def create_and_train_model(scaled_train_filename="sales_data_training_scaled.csv",
                        model_filename='train_model.h5', save_log=False, export_model=False):
    """
    Create neural network model and train it on train data. Save the model to file
    :param scaled_train_filename: str
            Filename containing scaled train data (.csv)

    :param model_filename: str
            Filename where the trained model will be stored (.h5)

    :param save_log: bool
            If True, save logs to visualize model in TensorBoard. Open TensorBoard in terminal with:
            tensorboard --logdir=logs

    :param export_model: bool
            If True, export model to TensorFlow

    """
    # Load scaled training data
    training_data_df = pd.read_csv(path + scaled_train_filename)

    # Drop target column
    X = training_data_df.drop('total_earnings', axis=1).values

    Y = training_data_df[['total_earnings']].values

    # Define the model
    model = Sequential()
    # Num neurons in the layer, num inputs (for first layer, as many as input features), type of activation function,
    # name for TensorBoard visualization
    model.add(Dense(50, input_dim=9, activation='relu', name='layer_1'))
    model.add(Dense(100, activation='relu', name='layer_2'))
    model.add(Dense(50, activation='relu', name='layer_3'))
    model.add(Dense(1, activation='linear', name='output_layer'))

    # Now create the model in Keras
    model.compile(loss='mse', optimizer='adam')

    if save_log:
        # Create a TensorBoard logger
        logger = keras.callbacks.TensorBoard(
            log_dir='logs_single_trial',
            write_graph=True,
            # histogram_freq=5, # for every 5 passes of training data
        )

        # Train the model
        model.fit(
            X,
            Y,
            epochs=50,
            shuffle=True,
            verbose=2,
            callbacks=[logger],
        )
    else:
        model.fit(
            X,
            Y,
            epochs=50,  # number of training times across the whole training set
            shuffle=True,
            verbose=2
        )

    # Save the model to disk
    model.save(path + model_filename)  # hdf5
    print('Model saved to disk!')

    if export_model:
        export_in_tf(model)


def export_in_tf(model, folder='exported_model'):
    """Export model to TensorFlow

    :param model: Keras model to export

    :param folder: str
            Folder where model is exported to
    """

    model_builder = tf.saved_model.builder.SavedModelBuilder(folder)

    # Create input and output in TensorFlow
    inputs = {
        'input': tf.saved_model.utils.build_tensor_info(model.input)
    }
    outputs = {
        'earnings': tf.saved_model.utils.build_tensor_info(model.output)
    }

    # Create signature
    signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    # Build TensorFlow model with current signature and structure
    model_builder.add_meta_graph_and_variables(
        K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        }
    )

    model_builder.save()


def evaluate_saved_model(scale_factor, subtract_factor, model_filename="train_model.h5",
                     scaled_test_filename="sales_data_test_scaled.csv",
                     to_predict_filename="04/proposed_new_product.csv"):
    """Load stored model and evaluate on test dataset, and predict for a new input

    :param scale_factor: float
                Divisive scaling factor to be used to scale data back

    :param add_factor: float
                Additive scaling factor to be used to scale data back

    :param model_filename: str
            Filename where Keras model is stored

    :param scaled_test_filename: str
            Filename with scaled test data

    :param to_predict_filename: str
            Filename with input data for which predicitons should be made (already scaled)
    """

    # Load the model and test dataset
    model = load_model(path + model_filename)
    test_data_df = pd.read_csv(path + scaled_test_filename)

    X_test = test_data_df.drop('total_earnings', axis=1).values
    Y_test = test_data_df[['total_earnings']].values

    test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
    print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

    # Load file on which predictions should be done - data are already scaled
    X_to_predict = pd.read_csv(path + to_predict_filename).values

    # Make a prediction with the NN
    prediction = model.predict(X_to_predict)

    # Grab first element
    prediction = prediction[0][0]

    # Re-scale the data from the 0-to-1 range back to dollars
    # These constants are from when the data was originally scaled down to the 0-to-1 range
    prediction = prediction - subtract_factor
    prediction = prediction / scale_factor

    print("Earnings Prediction for Proposed Product - ${}".format(prediction))


if __name__ == "__main__":

    filename_train = 'sales_data_training.csv'
    filename_test = 'sales_data_test.csv'

    scaled_train_filename, scaled_test_filename, scale_factor, subtract_factor = scale_data(filename_train, filename_test)
    create_and_train_model(scaled_train_filename, model_filename='train_model.h5', save_log=True, export_model=True)
    evaluate_saved_model(scale_factor, subtract_factor, model_filename='train_model.h5',
                         scaled_test_filename=scaled_test_filename)