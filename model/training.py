import torch
import model.load_data
import model.model as mdl
from tqdm import tqdm
import datetime
import wandb

def train_and_evaluate(csv_file, root_dir, probs_file, counts_file, num_classes, device, epochs=10):
    # Create a unique filename for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"

    with open(log_filename, "w") as log_file:
        # Load data with the updated function from load_data.py
        train_loader, test_loader = model.load_data.load_data(csv_file, root_dir, probs_file, counts_file, num_classes)

        # Initialize Weights and Biases
        wandb.init(project='human-anno-model-comparison', entity='mnkbone')  
        
        # Instantiate the model
        model = mdl.PooledMultinomialModel(device, num_classes)

        # Watch the PyTorch model inside the custom model class
        wandb.watch(model.model)

        # Train and evaluate the model
        model.train(train_loader, epochs, log_file)
        accuracy = model.evaluate(test_loader, log_file)

        log_file.write(f'Final Accuracy: {accuracy}%\n')

        # Log final accuracy to wandb
        wandb.log({'final_accuracy': accuracy})

        return accuracy
