from train_contrastive_model import train
from evaluate import evaluate
from config import Config

def main():
    print("Starting training...")
    train(Config)
    
    print("Evaluating model...")
    evaluate(Config)

if __name__ == '__main__':
    main()