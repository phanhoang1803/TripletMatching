class Config:
    # Model parameters
    feature_dim = 512
    context_input_dim = 100  # Adjust based on your context data
    context_hidden_dim = 256
    vocab_size = 10000  # Adjust based on your vocabulary
    embed_dim = 300
    caption_hidden_dim = 512
    ff_hidden_dim = 1024
    output_dim = 256

    # Training parameters
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    num_workers = 4

    # Data parameters
    train_data_path = 'path/to/train/data'
    val_data_path = 'path/to/val/data'
    test_data_path = 'path/to/test/data'

config = Config()