import mdshare
import numpy as np
import matplotlib.pyplot as plt
import argparse # Moduł ułatwia napisanie interfejsu wiersza poleceń łatwego w obsłudze.
from sklearn.manifold import TSNE  # Narzędzie do wizaualizacji nadych wielowymiarowych.
import yaml

parser = argparse.ArgumentParser() # Wyświetlanie komunikatu.
parser.add_argument('-s', '--step', metavar='', type=int, default=1000,
                    help="Set the step of the samples. Higher number means more samples will be ignored (1000 by default).")
parser.add_argument('-p', '--perplexity', metavar='', type=float, default=30.,
                    help="The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Different values can result in significanlty different results (30. by default).")
parser.add_argument('-e', '--early_exaggeration', metavar='', type=float, default=12.,
                    help="Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. For larger values, the space between natural clusters will be larger in the embedded space. Again, the choice of this parameter is not very critical. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high (12. by default).")
parser.add_argument('-l', '--learning_rate', metavar='', type=float, default=200.,
                    help="The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high, the data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the learning rate is too low, most points may look compressed in a dense cloud with few outliers. If the cost function gets stuck in a bad local minimum increasing the learning rate may help (200. by default).")
parser.add_argument('-n', '--n_iter', metavar='', type=int, default=1000,
                    help="Maximum number of iterations for the optimization. Should be at least 250 (1000 by default).")
parser.add_argument('-v', '--verbose', metavar='', type=int, default=0, help="Verbosity level (0 by default).")
parser.add_argument('-a', '--angle', metavar='', type=float, default=0.5,
                    help="The trade-off between speed and accuracy for Barnes-Hut T-SNE. ‘angle’ is the angular size of a distant node as measured from a point. If this size is below ‘angle’ then it is used as a summary node of all points contained within it. This method is not very sensitive to changes in this parameter in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing computation time and angle greater 0.8 has quickly increasing error (0.5 by default).")

args = parser.parse_args()
parsedStep = args.step
parsedPerplexity = args.perplexity
parsedEarly_exaggeration = args.early_exaggeration
parsedLearning_rate = args.learning_rate
parsedN_iter = args.n_iter
parsedVerbose = args.verbose
parsedAngle = args.angle


dataset = mdshare.fetch('alanine-dipeptide-3x250ns-heavy-atom-distances.npz') # ściąganie danych
with np.load(dataset) as f:
    X = np.vstack([f[key] for key in sorted(f.keys())])


Y = TSNE(n_components=2).fit_transform(X[  ::parsedStep])  # Dopasowywanie wielowymiarowych danych na wejściu do dwuwymiarowej tablicy.
plt.scatter(Y[:, 0], Y[:, 1], s=10) # tworzenie wykresu przyjmującego jako x pierwszą kolumnę i jako y drugą kolumnę danych Y
plt.axis('square') # wyrónwywanie długości osi
plt.show()



