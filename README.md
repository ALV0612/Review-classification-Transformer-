# BERT Classifier

## Installation

To use this project, you'll need to have the following dependencies installed:

- Python 3.7 or higher
- PyTorch 1.7.1 or higher
- PyTorch Lightning 1.4.2 or higher
- Transformers 4.11.3 or higher

You can install the required packages using pip:

```
pip install -r requirements.txt
```

## Usage

1. Mount your Google Drive to access the dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Prepare the dataset:

   - The dataset should be in a CSV file with the following columns:
     - `normalized_content`: The input text data.
     - `score`: The target labels.
   - Ensure the CSV file is located in the `/content/drive/MyDrive/GIt/` directory.

3. Run the BERT Classifier:

   ```python
   # Set the hyperparameters
   batch_size = 48
   max_length = 256
   num_workers = 1

   # Initialize the data module
   data_module = BERTDataModule(
       texts=texts,
       labels=labels,
       batch_size=batch_size,
       max_length=max_length,
       num_workers=num_workers
   )

   # Initialize the model
   model = BERTClassifier()
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)

   # Set up the trainer
   model_checkpoint = ModelCheckpoint(
       dirpath='/content/drive/MyDrive/GIt/checkpoint/',
       monitor="val_loss",
       verbose=True,
       mode="min",
       save_top_k=1
   )

   early_stopping = EarlyStopping(
       monitor="val_loss",
       mode="min",
       min_delta=1e-4,
       patience=5
   )

   callbacks = [model_checkpoint, early_stopping]

   trainer = pl.Trainer(
       max_epochs=10,
       detect_anomaly=True,
       callbacks=callbacks
   )

   # Train the model
   trainer.fit(model, data_module)
   ```

## API

The main components of the project are:

1. `BERTClassifier`: The PyTorch Lightning module that defines the BERT-based classification model.
2. `BERTDataModule`: The PyTorch Lightning data module that handles the dataset, tokenization, and data loading.
3. `CustomDataset`: The custom PyTorch dataset class that preprocesses the input text and labels.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open a new issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

