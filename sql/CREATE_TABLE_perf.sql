CREATE TABLE IF NOT EXISTS perf (
       -- identification
       id INTEGER PRIMARY KEY,
       name TEXT,
       -- hyperparameters
       PARAMETRIZATION TEXT,
       HIDDEN_L1 REAL,
       HIDDEN_L2 REAL,
       OUTPUT_L1 REAL,
       OUTPUT_L2 REAL,
       NLAYERS INT,
       NUNITS INT,
       LEARNING_RATE REAL,
       BATCH_NORM INT,
       DROPOUT_INPUT REAL,
       DROPOUT_HIDDEN REAL,
       BATCH_SIZE INT,
       NORMALIZATION TEXT,
       -- metrics
       N_EXCLUDED REAL,
       N_EXCLUDED_ABOVE_MBJ REAL
);
