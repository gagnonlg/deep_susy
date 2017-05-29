from model import ModelDefinition
from numpy.random import uniform, randint, choice

definition = ModelDefinition(
    name='GttNN_4',
    n_hidden_layers=randint(1, 10),
    n_hidden_units=randint(100, 1000),
    learning_rate=uniform(0.001, 0.1),
    momentum=uniform(0, 1),
    l2_reg=uniform(0, 1e-4),
    min_epochs=50,
    max_epochs=500,
    patience=10,
    reweight=choice([True, False]),
    normalize=choice([True, False]),
    reduceLR_factor=choice([1, uniform(0.1, 0.5)]),
    reduceLR_patience=5,
    early_stop_metric=choice(['loss', 'precision', 'recall', 'fmeasure'])
)
