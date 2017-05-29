from model import ModelDefinition
from numpy.random import uniform, randint, choice

definition = ModelDefinition(
    name='GttNN_2',
    n_hidden_layers=randint(1, 10),
    n_hidden_units=randint(100, 1000),
    learning_rate=round(uniform(0.001, 0.1), 3),
    momentum=choice([0, round(uniform(0, 1), 3)]),
    l2_reg=round(uniform(0, 1e-4), 7),
    min_epochs=50,
    max_epochs=500,
    patience=5,
    reweight=False,
    normalize=choice([True, False])
)
