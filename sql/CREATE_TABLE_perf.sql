CREATE TABLE IF NOT EXISTS perf (
       -- identification
       id INTEGER PRIMARY KEY,
       name TEXT,
       -- hyperparameters
       n_hidden_layers INT,
       n_hidden_units INT,
       normalization TEXT,
       l2 REAL,
       -- metrics
       n_excluded_training,
       n_excluded_validation
);
