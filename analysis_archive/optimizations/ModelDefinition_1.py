from model import ModelDefinition
from numpy.random import uniform, randint, choice

definition = ModelDefinition(
    name='GttNN_1',
    n_hidden_layers=randint(1, 7),
    n_hidden_units=randint(100, 301),
    learning_rate=round(uniform(0.001, 0.1), 3),
    momentum=choice([0, round(uniform(0, 1), 3)]),
    l2_reg=choice([0, round(uniform(1e-7, 1e-5), 5)]),
    min_epochs=50,
    max_epochs=500,
    patience=5,
    reweight=choice([True, False]),
    normalize=True
)
