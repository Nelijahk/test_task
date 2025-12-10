# paths to all files (it is expected them to be in the same directory as .py files)
TRAIN_DATA_PATH = "train.csv"
TEST_DATA_PATH = "hidden_test.csv"
OUTPUT_PREDICTIONS_PATH = "predictions.csv"

# hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001
RANDOM_STATE = 666

SCALER_FILENAME = "scaler.pkl"
MODEL_WEIGHTS_FILENAME = "reg_model_weights.pth"