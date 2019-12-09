import keras, logging, random, pydot, copy, uuid, os, sys, csv, json
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

basepath = "./"     #"/dbfs/FileStore/"

class HistoricalMarker:
    
    def __init__(self):
        self.module_counter = 0
        self.blueprint_counter = 0
        self.individual_counter = 0
    
    def mark_module(self):
        self.module_counter += 1
        return self.module_counter

    def mark_blueprint(self):
        self.blueprint_counter += 1
        return self.blueprint_counter
    
    def mark_individual(self):
        self.individual_counter += 1
        return self.individual_counter

class NameGenerator:
    def __init__(self):
        self.counter = 0
    
    def generate(self):
        self.counter += 1
        return self.counter

class ModuleComposition(Enum):
    INPUT = "input"
    INTERMED = "intermed"
    CONV = "conv2d"
    DENSE = "dense"
    OUTPUT = "output"
    COMPILER = "compiler"

class ComponentParameters(Enum):
    CONV2D = (keras.layers.Conv2D, {"filters": ([8, 16], 'int'), "kernel": ([3, 5, 7], 'list'), "stride": ([1, 2, 3], 'list')})
    MAXPOOLING2D = (keras.layers.MaxPooling2D, {"kernel":[3, 5, 7]})
    FLATTEN = (keras.layers.Flatten, 0)
    DENSE = (keras.layers.Dense, {"units":128, "activation":"relu"})

class Datasets:
    def __init__(self, complete=None, test=None, training=None, validation=None):
        self.complete=complete
        self.training=training
        self.validation=validation
        self.test=test
        self.custom_fit_args = None
        self.SAMPLE_SIZE = len(training[0])
        self.TEST_SAMPLE_SIZE = len(test[0])
    
    @property
    def shape(self):
        return self.complete.shape

    def split_complete(self):
        """
        Returns a full split from the complete dataset into training and validation
        """
        pass

class Component(object):
    """
    Represents a basic unit of a topology.
    keras_component: the keras component being represented
    """
    def __init__(self, representation, keras_component=None, complementary_component=None, keras_complementary_component=None, component_type=None):
        self.representation = representation
        self.keras_component = keras_component
        self.complementary_component = complementary_component
        self.keras_complementary_component = keras_complementary_component
        self.component_type = component_type

    def get_layer_size(self):
        if self.component_type == "conv2d":
            return self.representation[1]["filters"]
        if self.component_type == "dense":
            return self.representation[1]["units"]

class Module(object):
    """
    Represents a set of one or more basic units of a topology.
    components: the most basic unit of a topology.
    """
    def __init__(self, components:dict, layer_type:ModuleComposition=ModuleComposition.INPUT, mark=None, component_graph=None, parents=None):
        self.components = components
        self.component_graph = component_graph
        self.layer_type = layer_type
        self.mark = mark
        self.weighted_scores = [99,0]
        self.score_log = []
        self.species = None
        self.parents = parents
        self.use_count = 0
    
    def __getitem__(self, item):
        return self.components[item]

    def get_module_size(self):
        module_size = 0
        for node in self.component_graph.nodes():
            module_size += self.component_graph.nodes[node]["node_def"].get_layer_size()
        return module_size
    
    def get_kmeans_representation(self):
        node_count = len(self.component_graph.nodes())
        edge_count = len(self.component_graph.edges())
        module_size = self.get_module_size()
        scores = self.weighted_scores
        return node_count, edge_count, module_size #, scores[0], scores[1]

    def update_scores(self, scores):
        self.score_log.append(scores)

    def update_weighted_scores(self):

        if len(self.score_log) > 0:
            avg_test_loss = np.array(self.score_log)[:,0].mean()
            avg_test_acc = np.array(self.score_log)[:,1].mean()
            self.weighted_scores = [avg_test_loss, avg_test_acc]
        else:
            pass

    def simple_crossover(self, parent_2, mark):
        
        parent_1_graph = self.component_graph
        parent_1_graph_nodes = [node for node in parent_1_graph.nodes()]
        parent_2_graph = parent_2.component_graph
        parent_2_graph_nodes = [node for node in parent_2_graph.nodes()]
        child_graph = self.component_graph.copy()
        child_graph_nodes = [node for node in child_graph.nodes()]

        for n in range(len(child_graph_nodes)):
            parent_1_node = parent_1_graph.nodes[parent_1_graph_nodes[n]]["node_def"]
            if n < len(parent_2_graph.nodes()):
                parent_2_node = parent_2_graph.nodes[parent_2_graph_nodes[n]]["node_def"]
            else:
                parent_2_node = parent_2_graph.nodes[random.choice(parent_2_graph_nodes)]["node_def"]

            if parent_2_node.component_type == parent_1_node.component_type:
                chosen_node_def = random.choice([parent_1_node, parent_2_node])
                child_graph.nodes[child_graph_nodes[n]]["node_def"] = copy.deepcopy(chosen_node_def)
            else:
                pass
            
        #print(f"child_graph node defs: {[child_graph.nodes[node]['node_def'].representation for node in child_graph.nodes()]}")

        child = Module(None, layer_type=self.layer_type, mark=mark, component_graph=child_graph)

        return child

class Blueprint:
    """
    Represents a topology made of modules.
    modules: modules composing a topology.
    """
    def __init__(self, modules:List[Module], input_shape=None, module_graph=None, mark=None, parents=None):
        self.modules = modules
        self.input_shape = input_shape
        self.module_graph = module_graph
        self.mark = mark
        self.weighted_scores = [99,0]
        self.score_log = []
        self.species = None
        self.parents = parents
        self.use_count = 0

    def __getitem__(self, item):
        return self.modules[item]

    def get_blueprint_size(self):
        blueprint_size = 0
        for node in self.module_graph.nodes():
            blueprint_size += self.module_graph.nodes[node]["node_def"].get_module_size()
        return blueprint_size
    
    def get_kmeans_representation(self):
        node_count = len(self.module_graph.nodes())
        edge_count = len(self.module_graph.edges())
        blueprint_size = self.get_blueprint_size()
        scores = self.weighted_scores
        return node_count, edge_count, blueprint_size #, scores[0], scores[1]

    def update_scores(self, scores):
        self.score_log.append(scores)

        for node in self.module_graph.nodes():
            self.module_graph.nodes[node]["node_def"].update_scores(scores)
        
    def update_weighted_scores(self):

        if len(self.score_log) > 0:
            avg_test_loss = np.array(self.score_log)[:,0].mean()
            avg_test_acc = np.array(self.score_log)[:,1].mean()
            self.weighted_scores = [avg_test_loss, avg_test_acc]
        else:
            pass

    def simple_crossover(self, parent_2, mark):
        
        parent_1_graph = self.module_graph
        parent_1_graph_nodes = [node for node in parent_1_graph.nodes()]
        parent_2_graph = parent_2.module_graph
        parent_2_graph_nodes = [node for node in parent_2_graph.nodes()]
        child_graph = self.module_graph.copy()
        child_graph_nodes = [node for node in child_graph.nodes()]

        for n in range(len(child_graph_nodes)):
            parent_1_node = parent_1_graph.nodes[parent_1_graph_nodes[n]]["node_def"]
            if n < len(parent_2_graph.nodes()):
                parent_2_node = parent_2_graph.nodes[parent_2_graph_nodes[n]]["node_def"]
            else:
                parent_2_node = parent_2_graph.nodes[random.choice(parent_2_graph_nodes)]["node_def"]

            if parent_2_node.layer_type == parent_1_node.layer_type:
                chosen_node_def = random.choice([parent_1_node, parent_2_node])
                child_graph.nodes[child_graph_nodes[n]]["node_def"] = copy.deepcopy(chosen_node_def)
            else:
                pass

        #print(f"child_graph node defs: {[child_graph.nodes[node]['node_def'].mark for node in child_graph.nodes()]}")

        child = Blueprint(None, input_shape=self.input_shape, module_graph=child_graph, mark=mark)

        return child

class Species:
    """
    Represents a group of topologies with similarities.
    properties: a set of common properties defining a species.
    """
    def __init__(self, name=None, species_type=None, group=None, properties=None, starting_generation=None):
        self.name = name
        self.species_type = species_type
        self.group = group
        self.properties = properties
        self.starting_generation = starting_generation
        self.weighted_scores = [99,0]

    def update_weighted_scores(self):
        if len(self.group) > 0:
            weighted_scores = [item.weighted_scores for item in self.group if item.weighted_scores != [99,0]]

            if len(weighted_scores) > 0:
                avg_test_loss = np.array(weighted_scores)[:,0].mean()
                avg_test_acc = np.array(weighted_scores)[:,1].mean()
                self.weighted_scores = [avg_test_loss, avg_test_acc]
                logging.log(21, f"Updated weighted scores for species {self.name}: {self.weighted_scores} based on {weighted_scores}")
        
class Individual:
    """
    Represents a topology in the Genetic Algorithm scope.
    blueprint: representation of the topology.
    species: a grouping of topologies set according to similarity.
    birth: generation the blueprint was created.
    parents: individuals used in crossover to generate this topology.
    """
       
    def __init__(self, blueprint:Blueprint, compiler=None, birth=None, model=None, name=None):
        self.blueprint = blueprint
        self.compiler = compiler
        self.birth = birth
        self.model = model
        self.name = name
        self.scores = [0,0]

    def generate(self, save_fig=True, generation=""):
        """
        Returns the keras model representing of the topology.
        """

        logging.log(21, f"Starting assembling of blueprint {self.blueprint.mark}.")

        layer_map = {}
        module_graph = self.blueprint.module_graph

        if (save_fig):
            plt.tight_layout()
            plt.subplot(121)
            nx.draw(module_graph, nx.drawing.nx_agraph.graphviz_layout(module_graph, prog='dot'), with_labels=True, font_weight='bold', font_size=6)
            l,r = plt.xlim()
            plt.xlim(l-5,r+5)
            try:
                plt.savefig(f'{basepath}/images/gen{generation}_blueprint_{self.blueprint.mark}_module_level_graph_parent1id{self.blueprint.parents[0].mark}_parent2id{self.blueprint.parents[1].mark}.png', show_shapes=True, show_layer_names=True)
            except:
                plt.savefig(f"{basepath}/images/gen{generation}_blueprint_{self.blueprint.mark}_module_level_graph.png", format="PNG", bbox_inches="tight")
            plt.clf()

        assembled_module_graph = nx.DiGraph()
        output_nodes = {}

        for node in module_graph.nodes():
            assembled_module_graph = nx.union(assembled_module_graph, module_graph.nodes[node]["node_def"].component_graph, rename=(None, f'{node}-'))
            node_order = [node for node in nx.algorithms.dag.topological_sort(module_graph.nodes[node]["node_def"].component_graph)]
            output_nodes[node] = node_order[-1]
            #output_nodes[node] = (max(module_graph.nodes[node]["node_def"].component_graph.nodes()))

        for node in module_graph.nodes():
            for successor in module_graph.successors(node):
                assembled_module_graph.add_edge(f'{node}-{output_nodes[node]}', f'{successor}-0')

        if (save_fig):
            plt.tight_layout()
            plt.subplot(121)
            nx.draw(assembled_module_graph, nx.drawing.nx_agraph.graphviz_layout(assembled_module_graph, prog='dot'), with_labels=True, font_weight='bold', font_size=6)
            l,r = plt.xlim()
            plt.xlim(l-5,r+5)
            try:
                plt.savefig(f'{basepath}/images/gen{generation}_blueprint_{self.blueprint.mark}_component_level_graph_parent1id{self.blueprint.parents[0].mark}_parent2id{self.blueprint.parents[1].mark}.png', show_shapes=True, show_layer_names=True)
            except:
                plt.savefig(f"{basepath}/images/gen{generation}_blueprint_{self.blueprint.mark}_component_level_graph.png", format="PNG", bbox_inches="tight")
            plt.clf()

        logging.log(21, f"Generated the assembled graph for blueprint {self.blueprint.mark}: {assembled_module_graph.nodes()}")

        logging.info(f"Generating keras layers")

        #Adds Input layer
        logging.info(f"Adding the Input() layer")
        model_input = keras.layers.Input(shape=self.blueprint.input_shape)
        logging.log(21, f"Added {model_input}")

        #Garantees connections are defined in the correct order
        node_order = nx.algorithms.dag.topological_sort(assembled_module_graph)
        component_graph = assembled_module_graph

        # Iterate over the graph connecting keras layers
        for component_id in node_order:
            layer = []

            # Create a copy of the original layer so we don't have duplicate layers in the model in the future. Generates the keras layer now.
            component = copy.deepcopy(component_graph.nodes[component_id]["node_def"])
            component_def = component.representation[0]
            parameter_def = component.representation[1]
            component.keras_component = component_def(**parameter_def)
            component.keras_component.name = component.keras_component.name + "_" + uuid.uuid4().hex

            if component.complementary_component != None:
                component_def = component.complementary_component[0]
                parameter_def = component.complementary_component[1]
                component.keras_complementary_component = component_def(**parameter_def)
                component.keras_complementary_component.name = component.keras_complementary_component.name + "_" + uuid.uuid4().hex

            logging.log(21, f"{component_id}: Working on {component.keras_component.name}. Specs: {component.representation}")

            # If the node has no inputs, then use the Model Input as layer input
            if component_graph.in_degree(component_id) == 0:
                layer.append(component.keras_component(model_input))
                logging.log(21, f"{component_id}: Added {layer}")
                if component.complementary_component != None:
                    layer.append(component.keras_complementary_component(layer[0]))
                    logging.log(21, f"{component_id}: Added complement {layer}")
            
            # Else, if only one input, include it as layer input
            elif component_graph.in_degree(component_id) == 1:
                predecessors = [layer_map[predecessor_id] for predecessor_id in component_graph.predecessors(component_id)][0]
                logging.log(21, f"{component_id}: is {predecessors[0].name} conv2d: {'conv2d' in predecessors[0].name}. is {component.keras_component.name} dense: {'dense' in component.keras_component.name}")
                # If dense connecting to previous conv, flatten
                if "conv2d" in predecessors[0].name and "dense" in component.keras_component.name:
                    logging.info(f"Adding a Flatten() layer between {predecessors[0].name} and {component.keras_component.name}")
                    layer = [keras.layers.Flatten()(predecessors[-1])]
                    logging.log(21, f"{component_id}: Added {layer}")
                    predecessors = layer
                layer = [component.keras_component(predecessors[-1])]
                logging.log(21, f"{component_id}: Added {layer}")
                if component.complementary_component != None:
                    layer.append(component.keras_complementary_component(layer[0]))
                    logging.log(21, f"{component_id}: Added complement {layer}")
            
            # Else, if two inputs, merge them and use the merge as layer input
            elif component_graph.in_degree(component_id) == 2:
                predecessors = [layer_map[predecessor_id] for predecessor_id in component_graph.predecessors(component_id)]
                for predecessor in range(len(predecessors)):
                    logging.log(21, f"{component_id}: is {predecessors[predecessor][0].name} conv2d: {'conv2d' in predecessors[predecessor][0].name}. is {component.keras_component.name} dense: {'dense' in component.keras_component.name}")
                    if "conv2d" in predecessors[predecessor][0].name and "dense" in component.keras_component.name:
                        logging.info(f"Adding a Flatten() layer between {predecessors[predecessor][0].name} and {component.keras_component.name}")
                        layer = [keras.layers.Flatten()(predecessors[predecessor][-1])]
                        logging.log(21, f"{component_id}: Added {layer}")
                        predecessors[predecessor] = layer
                
                logging.info(f"Adding a Merge layer between {predecessors[0][0].name} and {predecessors[1][0].name}")
                merge_layer = keras.layers.concatenate([predecessors[0][-1], predecessors[1][-1]])
                logging.log(21, f"{component_id}: Added {merge_layer}")
                layer = [component.keras_component(merge_layer)]
                logging.log(21, f"{component_id}: Added {layer}")
                if component.complementary_component != None:
                    layer.append(component.keras_complementary_component(layer[-1]))
                    logging.log(21, f"{component_id}: Added complement {layer}")

            # Store model layers as a reference while the model is still not assembled 
            layer_map[component_id] = layer

        # Assemble model
        logging.log(21, layer_map)
        logging.log(21, f"Using input {model_input} and output {layer_map[max(layer_map)][-1]}")
        self.model = keras.models.Model(inputs=model_input, outputs=layer_map[max(layer_map)][-1])
        self.model.compile(**self.compiler)
        try:
            plot_model(self.model, to_file=f'{basepath}/images/gen{generation}_blueprint_{self.blueprint.mark}_layer_level_graph_parent1id{self.blueprint.parents[0].mark}_parent2id{self.blueprint.parents[1].mark}.png', show_shapes=True, show_layer_names=True)
        except:
            plot_model(self.model, to_file=f'{basepath}/images/gen{generation}_blueprint_{self.blueprint.mark}_layer_level_graph.png', show_shapes=True, show_layer_names=True)

    def fit(self, input_x, input_y, training_epochs=1, validation_split=0.15, current_generation="", custom_fit_args=None):
        """
        Fits the keras model representing the topology.
        """

        logging.info(f"Fitting one individual for {training_epochs} epochs")
        self.generate(generation=current_generation)
        if custom_fit_args is not None:
            fitness = self.model.fit_generator(**custom_fit_args)
        else:
            fitness = self.model.fit(input_x, input_y, epochs=training_epochs, validation_split=validation_split, batch_size=128)

        logging.info(f"Fitness for individual {self.name} using blueprint {self.blueprint.mark} after {training_epochs} epochs: {fitness.history}")

        return fitness

    def score(self, test_x, test_y):
        """
        Scores the keras model representing the topology.

        returns test_loss, test_acc
        """
        
        logging.info(f"Scoring one individual")
        scores = self.model.evaluate(test_x, test_y, verbose=1)
        
        #Update scores for blueprints (and underlying modules)
        self.blueprint.update_scores(scores)
        self.scores = scores

        logging.info(f"Test scores for individual {self.name} using blueprint {self.blueprint.mark} after training: {scores}")

        return scores

    def extract_blueprint(self):
        """
        Generates a Blueprint description of the self.model variable.
        """
        pass

class Population:
    """
    Represents the population containing multiple individual topologies and their correlations.
    """
    def __init__(self, datasets=None, individuals=[], blueprints=[], modules=[], hyperparameters=[], input_shape=None, population_size=1, compiler=None):
        self.datasets = datasets
        self.individuals = individuals
        self.blueprints = blueprints
        self.modules = modules
        self.historical_marker = HistoricalMarker()
        self.hyperparameters = hyperparameters
        self.module_species = None
        self.blueprint_species = None
        self.input_shape = input_shape
        self.population_size = population_size
        self.compiler = compiler

    def create_module_population(self, size, global_configs, possible_components, possible_complementary_components):
        """
        Creates a population of modules to be used in blueprint populations.
        Can be evolved over generations.
        """

        new_modules = []

        for n in range(size):
            mark = self.historical_marker.mark_module()
            new_module = GraphOperator().random_module(global_configs, 
                                                        possible_components, 
                                                        possible_complementary_components,
                                                        name=mark)
            new_module.mark = mark
            new_modules.append(new_module)

        #print(new_modules)
        self.modules = new_modules

    def create_blueprint_population(self, size, global_configs, possible_components, possible_complementary_components,
                                    input_configs, possible_inputs, possible_complementary_inputs,
                                    output_configs, possible_outputs, possible_complementary_outputs):
        """
        Creates a population of blueprints to be used in individual populations.
        Can be evolved over generations.
        """

        new_blueprints = []

        for n in range(size):

            mark = self.historical_marker.mark_blueprint()

            #Create a blueprint
            input_shape = self.input_shape
            new_blueprint = GraphOperator().random_blueprint(global_configs,
                                                            possible_components, 
                                                            possible_complementary_components, 
                                                            input_configs,
                                                            possible_inputs,
                                                            possible_complementary_inputs,
                                                            output_configs,
                                                            possible_outputs,
                                                            possible_complementary_outputs,
                                                            input_shape,
                                                            node_content_generator=self.return_random_module,
                                                            args={},
                                                            name=mark)

            new_blueprint.mark = mark
            new_blueprints.append(new_blueprint)

        #print(new_blueprints)
        self.blueprints = new_blueprints
    
    def create_individual_population(self, size=1, compiler=None):
        """
        Creates a population of individuals to be compared.
        Can be evolved over generations.
        """

        for item in self.individuals:
            if item.model != None:
                del item.model
            del item

        new_individuals = []

        K.clear_session()
        ## TODO: change the parameterization of the compiler variable.
        # Currently here because the Optimizer needs to be instantiated in every TFGraph crated, because K.clear_session deletes it.
        compiler = {"loss":"categorical_crossentropy", "optimizer":keras.optimizers.Adam(lr=0.005), "metrics":["accuracy"]}
        #compiler["optimizer"] = eval(compiler['optimizer'])

        for n in range(size):

            #Create a blueprint
            input_shape = self.input_shape
            new_blueprint = self.return_random_blueprint()

            #Create individual with the Blueprint
            new_individual = Individual(blueprint=new_blueprint, name=self.historical_marker.mark_individual(), compiler=compiler)
            new_individuals.append(new_individual)

        self.individuals = new_individuals

    def return_random_module(self, prefer_unused=True):
        unused = [module for module in self.modules if module.use_count == 0]
        use_counts = [module for module in self.modules]
        use_counts.sort(key=lambda x: (x.use_count), reverse=False)

        if unused != [] and prefer_unused:
            choice = random.choice(unused)
            choice.use_count += 1
            return choice
        elif prefer_unused:
            use_counts[0].use_count += 1
            return use_counts[0]
        else:
            choice = random.choice(self.modules)
            choice.use_count += 1
            return choice

    def return_random_blueprint(self, prefer_unused=True):
        unused = [blueprint for blueprint in self.blueprints if blueprint.use_count == 0]
        use_counts = [blueprint for blueprint in self.blueprints]
        use_counts.sort(key=lambda x: (x.use_count), reverse=False)

        if unused != [] and prefer_unused:
            choice = random.choice(unused)
            choice.use_count += 1
            return choice
        elif prefer_unused:
            use_counts[0].use_count += 1
            return use_counts[0]
        else:
            choice = random.choice(self.blueprints)
            choice.use_count += 1
            return choice

    def return_individual(self, name):
        for individual in self.individuals:
            if individual.name == name:
                return individual

        return False

    def return_best_individual(self):
        best_fitting = max(self.individuals, key=lambda indiv: (indiv.scores[1], -indiv.scores[0]))
        return best_fitting

    def apply_kmeans_speciation(self, items, n_clusters, species_type):
        """
        Apply KMeans to (re)start species
        """
        item_species = []
        representations = []

        for item in items:
            representations.append(item.get_kmeans_representation())
        
        classifier = KMeans(n_clusters=n_clusters, random_state=0)
        classifications = classifier.fit_predict(scale(representations))

        for species_name in range(len(classifier.cluster_centers_)):
            group = []
            for n in range(len(classifications)):
                if classifications[n] == species_name:
                    group.append(items[n])

            item_species.append(Species(name=species_name, species_type=species_type, group=group))

        for species in item_species:
            for item in species.group:
                item.species = species
        
        logging.log(21, f"KMeans generated {n_clusters} species using: {representations}.")
        
        return item_species, classifications

    def apply_centroid_speciation(self, items, species_list):
        """
        Apply the NearestCentroid method to assign members to existing species.
        Centroids are calculated based on existing species members and new members are assigned to the closest centroid.
        An accuracy threshold can be specified so new species are generated in case new members dont fit the existing centroid accordingly.
        """

        #Collect previous representations for centroid calculations.
        previous_member_representations = []
        previous_labels = []
        for species in species_list:
            previous_member_representations = previous_member_representations + [item.get_kmeans_representation() for item in species.group]
            previous_labels = previous_labels + [item.species.name for item in species.group]
        logging.log(21,(f"Previous species members: {previous_member_representations}. \nPrevious labels: {previous_labels}"))

        #Collect current representations for classification
        member_representations = []
        for item in items:
            member_representations.append(item.get_kmeans_representation())

        #Scale features using the whole data
        scaled_representations = scale(previous_member_representations + member_representations)
        #Select only speciated members to train the classifier
        scaled_previous_member_representations = scaled_representations[:len(previous_member_representations)]
        #Fit data to centroids
        classifier = NearestCentroid().fit(scaled_previous_member_representations, previous_labels)
        #Predict label to all data. New labels must be THE SAME as old labels, if they existed previously.
        all_classifications = classifier.predict(scaled_representations)
        logging.log(21,f"All Classifications: {all_classifications}")

        new_classifications = all_classifications[len(previous_member_representations):]
        logging.log(21,f"Old Classifications: {previous_labels}")
        logging.log(21,f"New Classifications: {new_classifications}")

        #Update species members
        for species in species_list:
            group = []
            for n in range(len(new_classifications)):
                if new_classifications[n] == species.name:
                    group.append(items[n])

            species.group = group

        #Update species info in items
        for species in species_list:
            for item in species.group:
                item.species = species

        return None

    def create_module_species(self, n_clusters):
        """
        Divides the modules in groups according to similarity.
        """

        module_species, module_classifications = self.apply_kmeans_speciation(self.modules, n_clusters, species_type="module")

        self.module_species = module_species

        logging.log(21, f"Created {n_clusters} module species.")
        for species in module_species:
            logging.log(21, f"Module species {species.name}: {[item.mark for item in species.group]}")

        return (module_classifications)

    def create_blueprint_species(self, n_clusters):
        """
        Divides the blueprints in groups according to similarity.
        """

        blueprint_species, blueprint_classifications = self.apply_kmeans_speciation(self.blueprints, n_clusters, species_type="blueprints")
        
        self.blueprint_species = blueprint_species

        logging.log(21, f"Created {n_clusters} blueprint species.")
        for species in blueprint_species:
            logging.log(21, f"Blueprint species {species.name}: {[item.mark for item in species.group]}")

        return (blueprint_classifications)
    
    def update_module_species(self, include_non_scored=True):
        """
        Divides the modules in groups according to similarity.
        """

        if (include_non_scored):
            modules = self.modules
        else:
            modules = [module for module in self.modules if module.weighted_scores != [99,0]]

        if (modules != []):
            self.apply_centroid_speciation(modules, self.module_species)
        else:
            logging.log(21, "Species not updated, no new scored modules.")

        return None
    
    def update_blueprint_species(self, include_non_scored=True):
        """
        Divides the blueprints in groups according to similarity.
        """

        if (include_non_scored):
            blueprints = self.blueprints
        else:
            blueprints = [blueprint for blueprint in self.blueprints if blueprint.weighted_scores != [99,0]]

        if (blueprints != []):
            self.apply_centroid_speciation(blueprints, self.blueprint_species)
        else:
            logging.log(21, "Species not updated, no new scored blueprints.")

        return None

    def crossover(self, items, species_list, marker_function, percent=0.2):
        """
        Generates new objects based on existing objects.
        """
        offspring = []
        exclusions = []
        
        for species in species_list:
            candidates = species.group
            if len(species.group) <= 1:
                pass
            else:
                #Choses a minimum of 2 or the highest even integer that fits under the percentage
                candidate_amount = max(2, int(len(species.group)*percent/2)*2)
                candidates = random.sample(species.group, k=candidate_amount)

                while len(candidates) > 0:
                    parent_1 = random.choice(candidates)
                    candidates.remove(parent_1)
                    parent_2 = random.choice(candidates)
                    candidates.remove(parent_2)
                    child = parent_1.simple_crossover(parent_2, marker_function())
                    child.parents = [parent_1, parent_2]

                    #Append the child to the population.
                    offspring.append(child)

                #Sort by scores
                species.group.sort(key=lambda x: (x.weighted_scores[1], -x.weighted_scores[0]), reverse=True)

                logging.log(21, f"Species {species.name} ordering: {([[item.mark, item.weighted_scores] for item in species.group])}")

                exclusions = exclusions + species.group[-int(candidate_amount/2):]

        return offspring, exclusions

    def crossover_modules(self, percent=0.2):
        """
        Generates new modules based on existing modules.
        """
        
        offspring, exclusions = self.crossover(self.modules, self.module_species, self.historical_marker.mark_module, percent=percent)
        
        preferable_exclusions = [item for item in self.modules if item.weighted_scores == [99,0]]

        for n in range(len(exclusions)):
            try:
                if n < len(preferable_exclusions):
                    logging.log(21, f"Excluding module: {preferable_exclusions[n].mark}")
                    self.modules.remove(preferable_exclusions[n])
                else:
                    logging.log(21, f"Excluding module: {exclusions[n].mark}")
                    self.modules.remove(exclusions[n])
            except:
                pass

        if offspring != [] and offspring != None:
            logging.log(21, f"Including offspring modules: {[item.mark for item in offspring]}")

        #Append the child to the population.
        self.modules = self.modules + offspring

    def crossover_blueprints(self, percent=0.2):
        """
        Generates new blueprints based on existing blueprints.
        """
        
        offspring, exclusions = self.crossover(self.blueprints, self.blueprint_species, self.historical_marker.mark_blueprint,  percent=percent)

        preferable_exclusions = [item for item in self.blueprints if item.weighted_scores == [99,0]]

        for n in range(len(exclusions)):
            try:
                if n < len(preferable_exclusions):
                    logging.log(21, f"Excluding blueprint: {preferable_exclusions[n].mark}")
                    self.blueprints.remove(preferable_exclusions[n])
                else:
                    preferable_exclusions = exclusions
                    logging.log(21, f"Excluding blueprint: {preferable_exclusions[n].mark}")
                    self.blueprints.remove(preferable_exclusions[n])
            except:
                pass
        
        for item in preferable_exclusions:
            del item

        if offspring != [] and offspring != None:
            logging.log(21, f"Including offspring blueprints: {[item.mark for item in offspring]}")

        #Append the offspring to the population.
        self.blueprints = self.blueprints + offspring
    
    def mutate_modules(self, percent, elitism_rate, possible_components, possible_complementary_components):
        """
        Mutates existing individuals.
        """

        args = {"possible_components": possible_components, "possible_complementary_components": possible_complementary_components}
        generator_function = GraphOperator().random_component

        mutation_variants = [GraphOperator().mutate_by_node_removal,
                             GraphOperator().mutate_by_node_addition_in_edges,
                             GraphOperator().mutate_by_node_addition_outside_edges,
                             GraphOperator().mutate_by_node_addition_in_edges,
                             GraphOperator().mutate_by_node_replacement]

        if len(self.modules) > 1:
            #Keep the 10% best intact
            candidates = self.modules
            candidates.sort(key=lambda x: (x.weighted_scores[1], -x.weighted_scores[0]), reverse=True)
            candidates = candidates[max(1, int(len(candidates)*elitism_rate)):]
            
            candidates = random.sample(candidates, k=round(len(self.modules)*percent))

            for candidate in candidates:
                mutation_operator = random.choice(mutation_variants)
                try:
                    mutated_graph = mutation_operator(candidate.component_graph, generator_function, args)
                    if (mutated_graph != None):
                        logging.log(21, f"Mutated candidate module {candidate.mark} with {mutation_operator}.")
                        candidate.component_graph = mutated_graph
                    else:
                        pass
                except:
                    logging.log(21, f"Impossible to mutate candidate module {candidate.mark} with {mutation_operator}.")

            logging.log(21, f"Mutated {len(candidates)} modules.")

        else:
            pass
    
    def mutate_blueprints(self, percent, elitism_rate, possible_components, possible_complementary_components):
        """
        Mutates existing individuals.
        """

        args = {}
        generator_function = self.return_random_module
        mutation_variants = [GraphOperator().mutate_by_node_removal,
                             GraphOperator().mutate_by_node_addition_in_edges,
                             GraphOperator().mutate_by_node_addition_outside_edges,
                             GraphOperator().mutate_by_node_addition_in_edges,
                             GraphOperator().mutate_by_node_replacement]

        if len(self.blueprints) > 1:
            #Keep the 10% best intact
            candidates = self.blueprints
            candidates.sort(key=lambda x: (x.weighted_scores[1], -x.weighted_scores[0]), reverse=True)
            candidates = candidates[max(1, int(len(candidates)*elitism_rate)):]
            
            candidates = random.sample(candidates, k=round(len(self.blueprints)*percent))

            for candidate in candidates:
                mutation_operator = random.choice(mutation_variants)
                try:
                    mutated_graph = mutation_operator(candidate.module_graph, generator_function, args)
                    logging.log(21, f"Mutated candidate blueprint {candidate.mark} to graph: {mutated_graph.nodes()} with {mutation_operator}.")
                    candidate.module_graph = mutated_graph
                except:
                    logging.log(21, f"Impossible to mutate candidate blueprint {candidate.mark} with {mutation_operator}.")

            logging.log(21, f"Mutated {len(candidates)} blueprints.")

        else:
            pass

    def update_shared_fitness(self):
        for module in self.modules:
            module.update_weighted_scores()
            module.score_log = []
        for blueprint in self.blueprints:
            blueprint.update_weighted_scores()
            blueprint.score_log = []
        for species in self.module_species:
            species.update_weighted_scores()
            species.score_log = []
        for species in self.blueprint_species:
            species.update_weighted_scores()
            species.score_log = []

    def reset_usage(self):
        for module in self.modules:
            if module.weighted_scores != [99,0]:
                 module.use_count == 1
            else:
                 module.use_count == 0
        for blueprint in self.blueprints:
            if blueprint.weighted_scores != [99,0]:
                 blueprint.use_count == 1
            else:
                 blueprint.use_count == 0                

    def iterate_fitness(self, training_epochs=1, validation_split=0.15, current_generation=0):
        """
        Fits the individuals and generates scores.

        returns a list composed of [invidual name, test scores, training history]
        """

        logging.info(f"Iterating fitness over {len(self.individuals)} individuals")
        iteration = []

        #(batch, channels, rows, cols)
        # Please murder me for this part I deserve it (random.sample doesn't work aaaah!!!)
        i = random.randint(0,59999-self.datasets.SAMPLE_SIZE)
        j = random.randint(0,9999-self.datasets.TEST_SAMPLE_SIZE)
        input_x = self.datasets.training[0][i:i+self.datasets.SAMPLE_SIZE]
        input_y = self.datasets.training[1][i:i+self.datasets.SAMPLE_SIZE]
        test_x = self.datasets.test[0][j:j+self.datasets.TEST_SAMPLE_SIZE]
        test_y = self.datasets.test[1][j:j+self.datasets.TEST_SAMPLE_SIZE]

        for individual in self.individuals:
            if (self.datasets.custom_fit_args is not None):
                history = individual.fit(input_x, input_y, training_epochs, validation_split, current_generation=current_generation, custom_fit_args=self.datasets.custom_fit_args)
            else:
                history = individual.fit(input_x, input_y, training_epochs, validation_split, current_generation=current_generation)
            score = individual.score(test_x, test_y)

            iteration.append([individual.name,
                              individual.blueprint.mark,
                              score,
                              individual.blueprint.get_kmeans_representation(),
                              (None if individual.blueprint.species == None else individual.blueprint.species.name),
                              current_generation])

        return iteration

    def iterate_generations(self, generations=1, training_epochs=1, validation_split=0.15, mutation_rate=0.5, crossover_rate=0.2, elitism_rate=0.1, possible_components=None, possible_complementary_components=None):
        """
        Manages generation iterations, applying the genetic algorithm in fact.

        returns a list of generation iterations
        """

        logging.info(f"Iterating over {generations} generations")
        iteration = None
        iterations = []
        best_scores = []
        csv_history = open(f"{basepath}iterations.csv", "w", newline="")
        csv_history.write("indiv,blueprint,scores,features,species,generation\n")
        csv_history.close()

        for generation in range(generations):
            logging.info(f" -- Iterating generation {generation} -- ")
            print(f" -- Iterating generation {generation} -- ")
            logging.log(21, f"Currently {len(self.modules)} modules, {len(self.blueprints)} blueprints, latest iteration: {iteration}")
            logging.log(21, f"Current modules: {[item.mark for item in self.modules]}")
            logging.log(21, f"Current blueprints: {[item.mark for item in self.blueprints]}")

            # Create representatives of blueprints
            self.create_individual_population(self.population_size, compiler=self.compiler)
            logging.log(21, f"Created individuals for blueprints: {[(item.name, item.blueprint.mark) for item in self.individuals]}")

            # Iterate fitness and record the iteration results
            iteration = self.iterate_fitness(training_epochs, validation_split, current_generation=generation)
            with open(f"{basepath}iterations.csv", "a", newline="") as csv_history:
                csv_history_writer = csv.writer(csv_history)
                csv_history_writer.writerows(iteration)
            iterations.append(iteration)
            logging.log(21, f"This iteration: {iteration}")

            # Update weighted scores of species
            self.update_shared_fitness()
            self.reset_usage()

            # Save best model
            # [name, blueprint_mark, score[test_loss, test_val], history]
            best_fitting = max(iteration, key=lambda x: (x[2][1], -x[2][0]))
            best_scores.append([f"generation {generation}", best_fitting])
            logging.log(21, f"Best model chosen: {best_fitting}")
            
            try:
                self.return_individual(best_fitting[0]).model.save(f"{basepath}/models/best_generation_{generation}.h5")
            except:
                logging.error(f"Model from generation {generation} could not be saved.")

            # Summarize execution
            self.summary(generation)

            # Crossover, mutate and update species.
            if generation != generations:
                self.crossover_modules(crossover_rate)
                self.mutate_modules(mutation_rate, elitism_rate, possible_components, possible_complementary_components)
                self.update_module_species()

                self.crossover_blueprints(crossover_rate)
                self.mutate_blueprints(mutation_rate, elitism_rate, possible_components, possible_complementary_components)
                self.update_blueprint_species()

        return best_scores

    def summary(self, generation=""):
        logging.log(21, f"\n\n --------------- Generation {generation} Summary -------------- \n")

        logging.log(21, f"Current {len(self.individuals)} individuals: {[item.name for item in self.individuals]}")
        
        # Summarize Blueprints
        blueprint_info = np.array2string(np.array([[item.mark] + item.weighted_scores + [None if item.species == None else item.species.name] + list(item.get_kmeans_representation()) for item in self.blueprints])).replace("\n", "").replace("] [", "] \n[")
        logging.log(21, f"Current {len(self.blueprints)} blueprints: \n[Mark, Test loss, Test acc, Species, Node Count, Edge Count, Neuron Count]\n{blueprint_info}")
        logging.log(21, f"Current {len(self.blueprint_species)} blueprint species: {[item.name for item in self.blueprint_species]}")
        logging.log(21, f"NOTE: Scores are considering past iteration. Not considering current blueprints for they are not yet all evaluated.")
        for species in self.blueprint_species:
            logging.log(21, f"Current blueprint species {species.name} scores: {species.weighted_scores}. members: {[item.mark for item in species.group]}")
            
        # Summarize Modules
        module_info = np.array2string(np.array([[item.mark] + item.weighted_scores + [None if item.species == None else item.species.name] + list(item.get_kmeans_representation()) for item in self.modules])).replace("\n", "").replace("] [", "] \n[")
        logging.log(21, f"Current {len(self.modules)} modules: \n[Mark, Test loss, Test acc, Species, Node Count, Edge Count, Neuron Count]\n{module_info}")
        logging.log(21, f"Current {len(self.module_species)} module species: {[item.name for item in self.module_species]}")
        logging.log(21, f"NOTE: Scores are considering past iteration. Not considering current modules for they are not yet all evaluated.")
        for species in self.module_species:
            logging.log(21, f"Current module species {species.name} scores: {species.weighted_scores}. members: {[item.mark for item in species.group]}")
        
        logging.log(21, f"\n -------------------------------------------- \n")

    def train_full_model(self, individual, training_epochs, validation_split, custom_fit_args=None):

        logging.info(f"Training {individual.name} for {training_epochs} epochs. Blueprint: {individual.blueprint.mark}.")
        
        #(batch, channels, rows, cols)
        input_x = self.datasets.training[0]
        input_y = self.datasets.training[1]
        test_x = self.datasets.test[0]
        test_y = self.datasets.test[1]

        history = individual.fit(input_x, input_y, training_epochs, validation_split, current_generation="final", custom_fit_args=custom_fit_args)
        score = individual.score(test_x, test_y)

        logging.info(f"Full model training scores: {score}")
        logging.info(f"Full model training history: {history.history}")

        individual.model.save(f"{basepath}/models/full_model_indiv{individual.name}_blueprint{individual.blueprint.mark}.h5")

        with open(f'{basepath}training.json', 'w', encoding='utf-8') as f:
            json.dump(history.history, f, ensure_ascii=False, indent=4)

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{basepath}/images/history_acc', show_shapes=True, show_layer_names=True)
        plt.clf()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{basepath}/images/history_loss', show_shapes=True, show_layer_names=True)

        return score, history

class GraphOperator:

    count=0

    @staticmethod
    def random_parameter_def(possible_parameters, parameter_name):
        parameter_def = {}
        parameter_values, parameter_type = possible_parameters[parameter_name]
        if parameter_type == 'int':
            parameter_def = random.randint(parameter_values[0], parameter_values[1])
        elif parameter_type == 'float':
            parameter_def = ((parameter_values[1] - parameter_values[0]) * random.random()) + parameter_values[0]
        elif parameter_type == 'list':
            parameter_def = random.choice(parameter_values)

        return parameter_def

    @staticmethod
    def save_graph_plot(filename, graph):
        plt.subplot(121)
        nx.draw(graph, nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot'), with_labels=True, font_weight='bold', font_size=6)
        plt.tight_layout()
        plt.savefig(f"{basepath}/images/{filename}", format="PNG", bbox_inches="tight")
        plt.clf()

    def random_component(self, possible_components, possible_complementary_components = None):
        component_type = random.choice(list(possible_components))
        component_def, possible_parameters = possible_components[component_type]

        parameter_def = {}
        for parameter_name in possible_parameters:
            parameter_def[parameter_name] = self.random_parameter_def(possible_parameters, parameter_name)

        if possible_complementary_components != None:
            compl_component_def, possible_compl_parameters = possible_complementary_components[random.choice(list(possible_complementary_components))]

            compl_parameter_def = {}
            for parameter_name in possible_compl_parameters:
                compl_parameter_def[parameter_name] = self.random_parameter_def(possible_compl_parameters, parameter_name)
            complementary_component = [compl_component_def, compl_parameter_def]
            keras_complementary_component = compl_component_def(**compl_parameter_def)
        else:
            complementary_component = None
            keras_complementary_component = None

        new_component = Component(representation=[component_def, parameter_def],
                                    keras_component=None,#component_def(**parameter_def),
                                    complementary_component=complementary_component,
                                    keras_complementary_component=None,#keras_complementary_component,
                                    component_type=component_type)
        return new_component

    def random_graph(self, node_range, node_content_generator, args=None):

        new_graph = nx.DiGraph()

        for node in range(node_range):
            node_def = node_content_generator(**args)
            new_graph.add_node(node, node_def=node_def)

            if node == 0:
                pass
            elif node > 0 and (node < node_range-1 or node_range <= 2):
                precedent = random.randint(0, node-1)
                new_graph.add_edge(precedent, node)
            elif node == node_range-1:
                leaf_nodes = [leaf_node for leaf_node in new_graph.nodes() if new_graph.out_degree(leaf_node)==0]
                root_node = min([node for node in new_graph.nodes() if new_graph.in_degree(node) == 0])
                leaf_nodes.remove(node)

                while (len(leaf_nodes) > 0):
                    if len(leaf_nodes) <= 2:
                        leaf_node = random.choice(leaf_nodes)
                        new_graph.add_edge(leaf_node, node)
                    else:
                        leaf_nodes.append(root_node)
                        random_node1 = random.choice(leaf_nodes)
                        simple_paths = [node for path in nx.all_simple_paths(new_graph, root_node, random_node1) for node in path]
                        leaf_nodes.remove(random_node1)
                        random_node2 = random.choice(leaf_nodes)
                        if (new_graph.in_degree(random_node2) >= 1 and random_node2 not in simple_paths and random_node2 != root_node):
                            new_graph.add_edge(random_node1, random_node2)
                    leaf_nodes = [leaf_node for leaf_node in new_graph.nodes() if new_graph.out_degree(leaf_node)==0]
                    leaf_nodes.remove(node)

        return new_graph

    def random_module(self, global_configs, possible_components, possible_complementary_components, name=0, layer_type=ModuleComposition.INTERMED):

        node_range = self.random_parameter_def(global_configs, "component_range")
        logging.log(21, f"Generating {node_range} components")
        print(f"Generating {node_range} components")

        graph = self.random_graph(node_range=node_range,
                                            node_content_generator=self.random_component,
                                            args = {"possible_components": possible_components,
                                                    "possible_complementary_components": possible_complementary_components})

        self.save_graph_plot(f"module_{name}_{self.count}_module_internal_graph.png", graph)
        self.count+=1
        new_module = Module(None, layer_type=layer_type, component_graph=graph)

        return new_module

    def random_blueprint(self, global_configs, possible_components, possible_complementary_components,
                        input_configs, possible_inputs, possible_complementary_inputs,
                        output_configs, possible_outputs, possible_complementary_outputs,
                        input_shape, node_content_generator=None, args={}, name=0):

        node_range = self.random_parameter_def(global_configs, "module_range")
        logging.log(21, f"Generating {node_range} modules")
        print(f"Generating {node_range} modules")

        if (node_content_generator == None):
            node_content_generator = self.random_module
            args = {"global_configs": global_configs,
                    "possible_components": possible_components,
                    "possible_complementary_components": possible_complementary_components}

        input_node = self.random_graph(node_range=1,
                                            node_content_generator=self.random_module,
                                            args = {"global_configs": input_configs,
                                                    "possible_components": possible_inputs,
                                                    "possible_complementary_components": None,
                                                    "layer_type": ModuleComposition.INPUT})
        #self.save_graph_plot(f"blueprint_{name}_input_module.png", input_node)

        intermed_graph = self.random_graph(node_range=node_range,
                                            node_content_generator=node_content_generator,
                                            args = args)
        #self.save_graph_plot(f"blueprint_{name}_intermed_module.png", intermed_graph)

        output_node = self.random_graph(node_range=1,
                                            node_content_generator=self.random_module,
                                            args = {"global_configs": output_configs,
                                                    "possible_components": possible_outputs,
                                                    "possible_complementary_components": possible_complementary_outputs,
                                                    "layer_type": ModuleComposition.OUTPUT})
        #self.save_graph_plot(f"blueprint_{name}_output_module.png", output_node)

        graph = nx.union(input_node, intermed_graph, rename=("input-", "intermed-"))
        graph = nx.union(graph, output_node, rename=(None, "output-"))
        graph.add_edge("input-0", "intermed-0")
        graph.add_edge(f"intermed-{max(intermed_graph.nodes())}", "output-0")
        self.save_graph_plot(f"blueprint_{name}_module_level_graph.png", graph)

        new_blueprint = Blueprint(None, input_shape, module_graph=graph)

        return new_blueprint

    def mutate_by_node_removal(self, graph, generator_function, args={}):

        new_graph = graph.copy()

        candidate_nodes = [node for node in new_graph.nodes() if new_graph.out_degree(node) > 0 and new_graph.in_degree(node) > 0]
        if len(candidate_nodes) == 0:
            return None

        selected_node = random.choice(candidate_nodes)
        
        # Removing a node
        if len(list(new_graph.predecessors(selected_node))) > 0 and len(list(new_graph.successors(selected_node))) > 0:
            predecessors = new_graph.predecessors(selected_node)
            successors = new_graph.successors(selected_node)

            new_edges = [(p,s) for p in predecessors for s in successors]

            new_graph.remove_node(selected_node)
            new_graph.add_edges_from(new_edges)

        return new_graph
    
    def mutate_by_node_addition_in_edges(self, graph, generator_function, args={}):

        new_graph = graph.copy()

        # "working around" to process the bad entity naming decisions I make in life.
        try:
            node = int(max(new_graph.nodes())) + 1
        except:
            node = "intermed-" + str(max([int(node.split('-')[1]) for node in new_graph.nodes() if 'input' not in node and 'output' not in node]+[0]) + 1)

        candidate_edges = [edge for edge in new_graph.edges()]
        if len(candidate_edges) == 0:
            return None

        selected_edge = random.choice(candidate_edges)
        
        # Adding a node 
        predecessor = selected_edge[0]
        successor = selected_edge[1]

        node_def = generator_function(**args)
        new_graph.add_node(node, node_def=node_def)
        new_graph.remove_edge(predecessor, successor)

        new_graph.add_edge(predecessor, node)
        new_graph.add_edge(node, successor)

        if len(leaf_nodes) != 1:
            return None

        return new_graph
    
    def mutate_by_node_addition_outside_edges(self, graph, generator_function, args={}):

        new_graph = graph.copy()

        try:
            node = int(max(new_graph.nodes())) + 1
        except:
            node = "intermed-" + str(max([int(node.split('-')[1]) for node in new_graph.nodes() if 'input' not in node and 'output' not in node]+[0]) + 1)

        node_def = generator_function(**args)

        # Select nodes that are not outputs
        candidate_predecessor_nodes = [node for node in new_graph.nodes() if new_graph.out_degree(node) > 0]

        #Select random predecessor
        predecessor = random.choice(candidate_predecessor_nodes)
        starting_node = min([node for node in new_graph.nodes() if new_graph.in_degree(node) == 0])
        simple_paths = [node for path in nx.all_simple_paths(new_graph, starting_node, predecessor) for node in path]

        # Select nodes that are not inputs and have at most 1 inputs (merge only supports 2 input layers)
        candidate_successor_nodes = [node for node in new_graph.nodes() if new_graph.in_degree(node) == 1 and node not in simple_paths]

        # If no successors available just create the node between an existing edge.
        if candidate_successor_nodes == []:
            successor = random.choice(new_graph.successors(predecessor))
            new_graph.remove_edge(predecessor, successor)
        else:
            successor = random.choice(candidate_successor_nodes)

        new_graph.add_node(node, node_def=node_def)
        new_graph.add_edge(predecessor, node)
        new_graph.add_edge(node, successor)

        # Only one leaf node allowed
        leaf_nodes = [leaf_node for leaf_node in new_graph.nodes() if new_graph.out_degree(leaf_node)==0]
        
        if len(leaf_nodes) != 1:
            return None

        return new_graph

    def mutate_by_node_replacement(self, graph, generator_function, args={}):

        new_graph = graph.copy()

        node_def = generator_function(**args)

        # Select nodes that are not outputs
        candidate_nodes = [node for node in new_graph.nodes() if 'input' not in node and 'output' not in node]

        if candidate_nodes == []:
            return None

        selected_node = random.choice(candidate_nodes)
        new_graph.nodes[selected_node]["node_def"] = node_def

        return new_graph