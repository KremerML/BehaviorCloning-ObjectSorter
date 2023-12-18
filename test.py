import unittest
import torch
import model.training as training

class TestTraining(unittest.TestCase):
    def test_train_evaluate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        csv_file = "G:\\GitHub\\BehaviorCloning-ObjectSorter\\data\\input\\cifar10h-raw.csv" #path to cifar10h-raw.csv
        root_dir = "G:\\GitHub\\BehaviorCloning-ObjectSorter\\data\\input\\cifar-10-images" #path to cifar-10-images
        probs_file = "G:\\GitHub\\BehaviorCloning-ObjectSorter\\data\\input\\cifar10h-probs.npy"  # Path to cifar10h-probs.npy
        counts_file = "G:\\GitHub\\BehaviorCloning-ObjectSorter\\data\\input\\cifar10h-counts.npy" # Path to cifar10h-counts.npy

        # Test with reduced epochs for quicker testing
        accuracy = training.train_and_evaluate(csv_file, root_dir, probs_file, counts_file, num_classes=10, device=device, epochs=1)
        self.assertTrue(0 <= accuracy <= 100)  # Check if accuracy is a valid percentage

if __name__ == '__main__':
    unittest.main()
