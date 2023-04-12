import time
import numpy as np
import pygame
from typing import Final, List, Tuple
import flappy_bird_gym
from argparse import ArgumentParser


class NNModel:
    def __init__(self, n_inputs: int, w_hidden: np.ndarray, w_output: np.ndarray, fitness=0):
        """
        NNModel's interface is neuron weights list based [[w_hidden][w_output]]
        except fitness record, it can not be changed after creation
        :param n_inputs: an integer suggest the number of inputs
        :param w_hidden: a 2 dimensional list holding weights for each hidden layer neuron
        :param w_output: a list holding weights for the output neuron
        """
        self.fitness: float = fitness
        self.n_inputs: Final[int] = n_inputs
        self.w_hidden: Final[np.ndarray] = w_hidden
        self.w_output: Final[np.ndarray] = w_output

    def __lt__(self, other):
        """
        used for comparing between two NNModels' fitness
        :param other: the NNModel to compare
        :return: True if this model is fitter
        """
        return self.fitness > other.fitness

    def __str__(self):
        return "fitness:{}\n{}\n{}".format(self.fitness, self.w_hidden, self.w_output)

    def get_stats(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        NNModel interface uses fitness, 2D array as hidden layer neuron weights and 1D array as output neuron weight
        :return: a tuple of all NNModel's properties
        """
        return self.fitness, self.w_hidden, self.w_output

    def get_genes(self) -> np.ndarray:
        """
        Genetic Algorithm Operates on gene (weight sequence)
        this method converts our Neural Network weights to a one dimensional array
        :return: 1 dimension array containing all weights
        """
        return np.concatenate((self.w_hidden.flatten(), self.w_output))

    def get_output(self, inputs: List[float]) -> float:
        """
        Using sigmoid activation function to produce a output
        calculate each neuron's output then activate it (including output_neuron)
        sigmoid (dot product of inputs & w_neuron)
        :return: the output corresponding to the inputs through NNModel, duh
        """

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        x_hidden = []
        for w_neuron in self.w_hidden:
            inactive_result = np.dot(w_neuron, inputs)
            x_hidden.append(sigmoid(inactive_result))

        return sigmoid(np.dot(x_hidden, self.w_output))

    def update_fitness(self, changes: float) -> None:
        """
        NNModel record its own fitness, fitness function is defined elsewhere however you wish
        :param changes: the change to our model fitness
        """
        self.fitness += changes

    @staticmethod
    def gene_to_model(genes: np.ndarray, n_inputs: int, n_hidden: int, fitness=0):
        """
        turning a gene sequence to NNModel instance
        :param genes: the gene sequence holding all weights for all neurons
        :param n_inputs: number of inputs this model takes
        :param n_hidden: number of hidden neurons
        :param fitness: the fitness record, by default is 0
        :return: a new model instance corresponding to the gene sequence
        """
        w_hidden: np.ndarray = np.array(genes[0: n_inputs * n_hidden]).reshape(-1, 2)
        w_output: np.ndarray = np.array(genes[n_inputs * n_hidden:])
        return NNModel(n_inputs, w_hidden, w_output, fitness)


class GeneticNN:
    def __init__(self, n_inputs: int, n_hidden: int, n_pop: int, weight_bounds: Tuple[int, int]):
        # np.random.seed(1)  # ONLY USE FOR DEVELOPING STAGE
        """
        Genetic Algorithm Operations here is gene (weights sequence) based: [float]
        excepts populations, it can not be changed after created

        when creating a new population
        we assign each model's initial weight randomly from a uniform distribution between weight_bounds

        :param n_inputs: number of inputs for Neural Network Model
        :param n_hidden: number of neurons in hidden layer (this model only support one hidden layer)
        :param n_pop: the size of population
        :param weight_bounds: the initial weights of all models uniform bound in tuple [lower, upper]
        """
        self.n_inputs: Final[int] = n_inputs
        self.n_pop: Final[int] = n_pop
        self.n_hidden: Final[int] = n_hidden
        self.n_genes: Final[int] = self.n_inputs * self.n_hidden
        self.population: List[NNModel] = []
        self.generation = 0

        for _ in range(n_pop):
            w_hidden = np.random.uniform(
                weight_bounds[0], weight_bounds[1], size=(self.n_hidden, self.n_inputs)
            )

            w_output = np.random.uniform(
                weight_bounds[0], weight_bounds[1], size=self.n_hidden
            )

            self.population.append(NNModel(n_inputs, w_hidden, w_output))

    def __str__(self):
        result: str = ""
        for model in self.population:
            result += (model.__str__() + "\n\n")
        return result

    def get_stats(self) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """
        get each model's property in its native way, packed into a list
        :return: a list of tuples containing each model's meta info
        """
        result: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for model in self.population:
            result.append(model.get_stats())
        return result

    def get_output(self, inputs: List[List[float]]) -> List[float]:
        """
        get output from every model in population
        :param inputs: a list of inputs corresponding to each model in population
        :return: a list of outputs
        """
        outputs: List[float] = []
        for i in range(self.n_pop):
            outputs.append(self.population[i].get_output(inputs[i]))
        return outputs

    def update_fitness(self, change: List[float]) -> None:
        """
        why are you even reading me -_-b
        this function update the entire population's fitness record with ease
        :param change: a list of change in fitness
        :return:
        """
        for i in range(self.n_pop):
            self.population[i].update_fitness(change[i])

    def get_best_fitness(self) -> float:
        """
        return the best fitness within the current population at the moment when this method is called
        :return: the best among us all (the best fitness at call time)
        """
        snapshot = self.population.copy()
        snapshot.sort()
        return snapshot[0].fitness

    def select(self, mode: str, n_winner: int) -> List[np.ndarray]:
        """
        used in re-populate, one of the 3 steps: select, crossover, mutate
        after sorting all the models by their fitness, select n_winner from the population's best half

        currently 3 mode provided
        1. best: return the top n_winners
        2. roulette: the top half populations as candidates, randomly select n_winners from them
           the fitter the candidate is, the higher probability it may get picked
        3. random: the top half population as candidates, randomly pick n_winner

        hopefully this will fix the issue after one generation all models became the same
        :return: a list of winner models
        """
        best_half = self.population.copy()
        best_half.sort()
        best_half = best_half[0: int(self.n_pop / 2)]

        selected_models: List[NNModel] = []

        if mode == "best":
            selected_models = best_half[0:n_winner]

        if mode == "roulette":
            # calculate each model's probability of being selected
            all_fitness = []
            for model in best_half:
                all_fitness.append(model.fitness)
            sum_fitness = np.sum(all_fitness)

            probabilities = []
            for n in all_fitness:
                probabilities.append(n / sum_fitness)
            selected_models = np.random.choice(best_half, n_winner, replace=False, p=probabilities)

        if mode == "random":
            selected_models = np.random.choice(best_half, 2, replace=False)

        selected_genes: List[np.ndarray] = []
        for model in selected_models:
            selected_genes.append(model.get_genes())

        return selected_genes

    @staticmethod
    def crossover(parents: List[np.ndarray], k_crossover) -> np.ndarray:
        """
        baby making :D
        we are using k-point crossover here
        what's k-point crossover? read wikipedia
        :param k_crossover: the number of crossover point
        :param parents: a tuple containing two parents' gene sequence
        :return: a offspring gene sequence after k-point crossed
        """
        # randomly select k cross point and create cross pattern
        choice_pool = list(range(1, len(parents[0]) - 1))
        cross_point = np.random.choice(choice_pool, k_crossover, replace=False)
        cross_point.sort()

        offspring_gene: List[float] = []
        is_a_gene: bool = True
        for i in range(len(parents[0])):
            if i in cross_point:
                is_a_gene = not is_a_gene
            offspring_gene.append(parents[0][i] if is_a_gene else parents[1][i])

        return np.array(offspring_gene)

    @staticmethod
    def mutation(genes: np.ndarray, n_mutations: int, weight_bounds: Tuple[float, float]) -> np.ndarray:
        """
        at few random positions, a completely new weight is replaced on the gene sequence position
        :param genes: a gene sequence to mutate
        :param n_mutations: number of gene snipe to replace
        :param weight_bounds: the boundary of random weight
        :return: a brand new gene sequence (somewhat new) :D
        """
        choice_pool = list(range(1, len(genes) - 1))
        mutate_point = np.random.choice(choice_pool, n_mutations, replace=False)
        for i in mutate_point:
            genes[i] = np.random.uniform(weight_bounds[0], weight_bounds[1])

        return genes

    def new_population(self) -> None:
        """
        reproduce a new population based on the current one
        first select the top half winner, they get to live to next generation (50%)
        then generate the offsprings by
            1. select best two parents for crossover (5%)
            2. select roulette few pairs parents for crossover (25%)
            3. select random pairs parents for crossover (15%)
            4. select random parents directly as offsprings (5%)
        mutate all offsprings at 4 random gene position
        go nuts
        :return: the excitement
        """
        # 50% original winners
        winners = self.select("best", int(self.n_pop / 2))

        offsprings: List[np.ndarray] = []

        # 5% best winners winners
        for _ in range(int(self.n_pop * 0.05)):
            offsprings.append(self.crossover(self.select("best", 2), 3))

        # 25% roulette winners offspring
        for _ in range(int(self.n_pop * 0.25)):
            offsprings.append(self.crossover(self.select("roulette", 2), 3))

        # 15% random winners offsprings
        for _ in range(int(self.n_pop * 0.15)):
            offsprings.append(self.crossover(self.select("random", 2), 3))

        # 5% random winners direct copy (later with mutation)
        for _ in range(int(self.n_pop * 0.05)):
            offsprings = offsprings + self.select("random", 2)

        # mutate all offsprings
        for baby in offsprings:
            baby = self.mutation(baby, 4, (-1, 1))

        # turn all genes into NNModels
        new_population: List[NNModel] = []
        for gene in winners:
            new_population.append(NNModel.gene_to_model(gene, self.n_inputs, self.n_hidden, 0))
        for gene in offsprings:
            new_population.append(NNModel.gene_to_model(gene, self.n_inputs, self.n_hidden, 0))
        self.population = new_population
        self.generation = self.generation + 1


def start(show_prints=False, show_gui=False, fps=60, agents_num=100):
    # create gym
    env = flappy_bird_gym.make("FlappyBird-v0")
    # get initial observation
    obs = env.reset(agents_num)

    # if GUI enabled, initialise the window and clear events
    if show_gui:
        env.render()
        pygame.event.pump()

    # create Genetic Algorithm Population with NN models
    GANN = GeneticNN(2, 10, agents_num, (-10, 10))
    GOAT = 0

    # Training, record fitness, and play
    while True:
        # to flap or not to flap
        predictions = GANN.get_output(obs)
        actions = []
        for x in predictions:
            actions.append(x > 0.5)

        # Processing the action:
        obs, reward, done, scores = env.step(actions)

        # record fitness
        GANN.update_fitness(reward)
        GOAT = GANN.get_best_fitness() if GANN.get_best_fitness() > GOAT else GOAT

        # if all agents died, start new population
        if all(done):
            print("\nGeneration {} Completed: Generation best: {}\n"
                  .format(GANN.generation, GANN.get_best_fitness()))
            GANN.new_population()
            obs = env.reset(agents_num)

        # logging
        if show_prints:
            print("\rCurrent Generation: {} \t GOAT: {}".format(GANN.generation, GOAT), end='', flush=True)

        # Rendering the game:
        if show_gui:
            env.render()
            time.sleep(1 / fps)  # FPS

    # clean up though not reachable right now lel
    # env.close()
    # return scores


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-g",
                        dest="show_gui",
                        action="store_true",
                        help="Whether the game GUI should be shown or not")

    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        action="store_true",
                        help="Print information while playing the game")

    parser.add_argument("-fps",
                        dest="fps",
                        type=int,
                        default=60,
                        help="Specify in how many FPS the game should run")

    parser.add_argument("-n",
                        dest="agents_num",
                        type=int,
                        default=100,
                        help="Specify in how many agents the game should has")

    options = parser.parse_args()

    start(show_prints=options.verbose, show_gui=options.show_gui, fps=options.fps, agents_num=options.agents_num)
