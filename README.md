# Disaster Response Pipeline Project

## Introduction
The Disaster Response Project uses a machine learning pipeline to categorize messages received during disaster events into appropriate categories in order for organizations to direct resources and responses.

## Project Structure
```
├── app/
│   ├── templates/       # HTML code for the web app
│   └── run.py           # Flask app script and graph setup
│   
│
├── data/
│   ├── DisasterResponse.db    # SQLite database containing the data
│   └── process_data.py        # Data cleaning and processing script
│
├── models/
│   ├── classifier.pkl         # The NLP model
│   └── train_classifier.py    # NLP Pipeline training script
│
├── pyproject.toml      # project dependencies
└── README.md           # Project documentation
```

## Data
The data used in this project is sourced from disaster response messages provided by [Figure Eight](https://www.figure-eight.com/). The dataset includes:

- **Messages**: Text messages in multiple languages.
- **Labels**: 36 categories indicating the type of aid needed.

## Model Details

### Preprocessing
This section is mainly handled in the `process_data.py` script.
- Languange categorization
- Tokenization and lemmatization
- Stopword removal
- Text vectorization using TF-IDF

### Model build
This section is mainly handled in the `train_classifier.py` script.

#### Model description
The classification model is built using:
- **TF-IDF** - Measure the importance of key words in order to classify a message
- **Message length** - Check i the length of the message affects the classification
- **Starting verb detection** - In order to assess if action is requested or urged

The main model used is a `random forest classifier`, adjusted adjusted using a `multi output classifier` since we have a set of categories as dependent variables in the model.

#### Pipeline
- The model uses the datatransformer Pipeline in order to perform the traning. 
[Pipeline](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html#) 

#### GridSeach
- The traning currently only uses two parameters for optimizing the model performance:
    - `features__text_features__vect__max_features`:
        * Asseses the number of features used by the TF-IDF vectorizer process to balance data usage and output. The current model tested between 5000 and 7500 terms, but this can be increased if more computational power is available.
    - `clf__estimator__class_weight`:
        * This parameter tests wether a weight is need for classes which are less present are not overlooked by the model. 
        * In our case we have several variables that have very few positive observations in the training data, and these may need to have a higher weight assigned to them.


- For improved optimization further parameter testing is adviced.

### Assessment
- The model has been assessed using precision, recal and f1-score for each separate class.
The overall assessment is that the model still needs improving and training, **especially when locking at recall for classes where there are few positive observations in the training data**.


## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/disaster-response-project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd disaster-response-project
   ```
3. Install the required Python dependencies (if using poetry) by running the command:
   ```bash
   poetry install
   ```

## Usage
1. Navigate to the project directory:
   ```bash
   cd disaster-response-project
   ```

2. To run ETL pipeline that cleans data and stores in database
    ```bash
     python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    ```
        
3. To run ML pipeline that trains classifier and saves
    ```bash
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```

4. Run the following command in the app's directory to run your web app.
    ```bash
    python run.py
    ```

5. Go to:
    ```bash
    http://0.0.0.0:3001/
    ```

## Acknowledgments
- [Udacity](https://www.udacity.com/) for providing the course content.
- [Figure Eight](https://www.figure-eight.com/) for the dataset.

## Additional sources
Discussions on testing a weights
https://medium.com/analytics-vidhya/optimize-hyperparameters-with-gridsearch-d351b0fd339d

Lemmatization
https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
