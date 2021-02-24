"""
configuration for experiments and analysis
"""
import os
from collections import OrderedDict

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skmultilearn.adapt import MLkNN
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.problem_transform import ClassifierChain, LabelPowerset, BinaryRelevance

PATH = os.path.join(os.path.dirname(__file__), "..", "data/").replace("""\\""", '/')
RAW_DATASET_PATH, CLEAN_DATASET_PATH = PATH + 'raw_dataset/', PATH + 'clean_dataset/'
MODELS_PATH, FIG_PATH = PATH + 'output/saved_models/', PATH + 'output/plotting/'

HOST_LIST = ('SRV', 'GW1', 'GW11', 'GW111')
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
save = False
ROUND = 4
SEED = 123
K_FOLD = 5
TEST_SIZE = 0.25
AVERAGE = 'macro'  # 'macro'  # 'micro'
R = 1.61803398875
CV_2 = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[TEST_SIZE, 1.0 - TEST_SIZE])
CV = IterativeStratification(n_splits=K_FOLD, order=2)
K = 3
C = 10
scaler = StandardScaler()

CLASSIFIERS = OrderedDict()
base_clfs = OrderedDict({
    "Neural Net": MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver='adam', alpha=0.0001, max_iter=200,
                                batch_size=1000, learning_rate="constant", learning_rate_init=0.01, max_fun=15000,
                                early_stopping=True, n_iter_no_change=50, random_state=SEED)
})

for clf_name, clf in base_clfs.items():
    CLASSIFIERS["CC [" + clf_name + "]"] = make_pipeline(scaler, ClassifierChain(clf))
    CLASSIFIERS["BR [" + clf_name + "]"] = make_pipeline(scaler, BinaryRelevance(clf))
    CLASSIFIERS["LP [" + clf_name + "]"] = make_pipeline(scaler, LabelPowerset(clf))

CLASSIFIERS["ML-kNN"] = make_pipeline(scaler, MLkNN(k=K))