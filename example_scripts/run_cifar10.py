import sys
sys.path.append("..")

import keras, logging, random, pydot, copy, uuid, os, csv, json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import List
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import scale
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import regularizers
import imp

kerascodeepneat = imp.load_source("kerascodeepneat", "./base/kerascodeepneat.py")

def run_cifar10_full(generations, training_epochs, population_size, blueprint_population_size, module_population_size, n_blueprint_species, n_module_species, final_model_training_epochs):
    from keras.datasets import cifar10

    #Set parameter tables
    global_configs = {
        "module_range" : ([1, 3], 'int'),
        "component_range" : ([1, 3], 'int')
    }
    input_configs = {
    "module_range" : ([1, 1], 'int'),
    "component_range" : ([1, 1], 'int')
    }
    output_configs = {
        "module_range" : ([1, 1], 'int'),
        "component_range" : ([1, 1], 'int')
    }

    possible_components = {
        "conv2d": (keras.layers.Conv2D, {"filters": ([16,48], 'int'), "kernel_size": ([1, 3, 5], 'list'), "strides": ([1], 'list'), "data_format": (['channels_last'], 'list'), "padding": (['same'], 'list'), "activation": (["relu"], 'list')}),
        #"dense": (keras.layers.Dense, {"units": ([8, 48], 'int')})
    }
    possible_inputs = {
        "conv2d": (keras.layers.Conv2D, {"filters": ([16,64], 'int'), "kernel_size": ([1], 'list'), "activation": (["relu"], 'list')})
    }
    possible_outputs = {
        "dense": (keras.layers.Dense, {"units": ([32,256], 'int'), "activation": (["relu"], 'list')})
    }

    possible_complementary_components = {
        #"maxpooling2d": (keras.layers.MaxPooling2D, {"pool_size": ([2], 'list')}),
        "dropout": (keras.layers.Dropout, {"rate": ([0, 0.5], 'float')})
    }
    possible_complementary_inputs = None
    possible_complementary_outputs = {
        "dense": (keras.layers.Dense, {"units": ([10,10], 'int'), "activation": (["softmax"], 'list')})
    }

    
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    num_classes = 10

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    validation_split = 0.15
    #training
    batch_size = 128

    #data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        )
    datagen.fit(x_train)

    my_dataset = kerascodeepneat.Datasets(training=[x_train, y_train], test=[x_test, y_test])
    my_dataset.SAMPLE_SIZE = 20000
    my_dataset.TEST_SAMPLE_SIZE = 2000

    logging.basicConfig(filename='execution.log',
                        filemode='w+', level=logging.INFO,
                        format='%(levelname)s - %(asctime)s: %(message)s')
    logging.addLevelName(21, "TOPOLOGY")

    logging.warning('This will get logged to a file')
    logging.info(f"Hi, this is a test run.")

    compiler = {"loss":"categorical_crossentropy", "optimizer":"keras.optimizers.Adam(lr=0.005)", "metrics":["accuracy"]}

    # Set configurations for full training session (final training)
    es = EarlyStopping(monitor='val_acc', mode='auto', verbose=1, patience=15)
    mc = ModelCheckpoint('best_model_checkpoint.h5', monitor='val_accuracy', mode='auto', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('training.csv')
    custom_fit_args = {"generator": datagen.flow(x_train, y_train, batch_size=batch_size),
    "steps_per_epoch": x_train.shape[0] // batch_size,
    "epochs": training_epochs,
    "verbose": 1,
    "validation_data": (x_test,y_test),
    "callbacks": [es, csv_logger]
    }                        
    improved_dataset = kerascodeepneat.Datasets(training=[x_train, y_train], test=[x_test, y_test])
    improved_dataset.custom_fit_args = custom_fit_args
    my_dataset.custom_fit_args = None

    # Initiate population
    population = kerascodeepneat.Population(my_dataset, input_shape=x_train.shape[1:], population_size=population_size, compiler=compiler)
  
    # Start with random modules
    population.create_module_population(module_population_size, global_configs, possible_components, possible_complementary_components)
    population.create_module_species(n_module_species)

    # Start with random modules
    population.create_blueprint_population(blueprint_population_size,
                                            global_configs, possible_components, possible_complementary_components,
                                            input_configs, possible_inputs, possible_complementary_inputs,
                                            output_configs, possible_outputs, possible_complementary_outputs)
    population.create_blueprint_species(n_blueprint_species)

    # Iterate generating, fitting, scoring, speciating, reproducing and mutating.
    iteration = population.iterate_generations(generations=generations,
                                                training_epochs=training_epochs,
                                                validation_split=validation_split,
                                                mutation_rate=0.5,
                                                crossover_rate=0.2,
                                                elitism_rate=0.1,
                                                possible_components=possible_components,
                                                possible_complementary_components=possible_complementary_components)

    print("Best fitting: (Individual name, Blueprint mark, Scores[test loss, test acc], History).\n", (iteration))

    # Return the best model
    best_model = population.return_best_individual()

    # Set data augmentation for full training
    population.datasets = improved_dataset
    print("Using data augmentation.")

    try:
        print(f"Best fitting model chosen for retraining: {best_model.name}")
        population.train_full_model(best_model, final_model_training_epochs, validation_split, custom_fit_args)
    except:
        population.individuals.remove(best_model)
        best_model = population.return_best_individual()
        print(f"Best fitting model chosen for retraining: {best_model.name}")
        population.train_full_model(best_model, final_model_training_epochs, validation_split, custom_fit_args)
  
if __name__ == "__main__":

    generations = 2
    training_epochs = 2
    final_model_training_epochs = 2
    population_size = 1
    blueprint_population_size = 10
    module_population_size = 30
    n_blueprint_species = 3
    n_module_species = 3

    def create_dir(dir):
        if not os.path.exists(os.path.dirname(dir)):
            try:
                os.makedirs(os.path.dirname(dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    create_dir("models/")
    create_dir("images/")
    
    run_cifar10_full(generations, training_epochs, population_size, blueprint_population_size, module_population_size, n_blueprint_species, n_module_species, final_model_training_epochs)