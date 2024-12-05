# Disaster Response Pipeline Project

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Data](#data)
4. [Model Details](#model-details)
   - 4.1 [Preprocessing](#preprocessing)
   - 4.2 [Model Build](#model-build)
     - 4.2.1 [Model Description](#model-description)
     - 4.2.2 [Pipeline](#pipeline)
     - 4.2.3 [GridSearch](#gridsearch)
   - 4.3 [Assessment](#assessment)
5. [Usage](#usage)
   - 5.1 [Installation](#installation)
   - 5.2 [Initiate Model](#initiate-model)
   - 5.3 [Web App](#web-app)
6. [Further Development](#further-development)
7. [Acknowledgments](#acknowledgments)
8. [Additional Sources](#additional-sources)

## Introduction
The Disaster Response Project uses a machine learning pipeline to categorize messages received during disaster events into appropriate categories in order for organizations to direct resources and responses.

## Project Structure
```
├── app/
│   ├── templates/       # HTML code for the web app
│   └── run.py           # Web app script and graph setup
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


## Usage

### Installation
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

### Initiate model 
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

### Web app
The app has two main functions. 

1. **Text categorization** - For categorizing a text message, enter the message in the `Enter to classify` field and push the `Classify Message` button. The selected text classifiers will appear below the under the Result text.

    - E.g. if you text "we are starving and are also lacking both drinking water and medicine" the categories "Related", "Aid Related" and "Water" will appear.

2. **Training data distribution** - If you follow the link that says `Training data distribution` this will take you to a set of graphs showing the distribution.
    
    - **Overall distribution**
        
        The first graph shows the overall distribution of the type of texts used by the model, direct messages, news and social media inputs.

    - **Direct message distribution**

        The second two graphs shows the distribtion of the direct messages, since these are the ones to be classified in the end.
        - The first of these shows the distribution of languages used. This could be used further for classification modeling in the future.
        - The second distribution shows how large a share of all messages that has had a positive observation in any of the categories.

## Further development
The results are still highly unbalanced and need further improvements too handle the many categories that are under represented.

Beyond the actual improvment of the NLP modell, one such action could be to look over the categories themselves and group them in accordance to the actions needed to be taken within them. This could also help with the balance problem.

## Acknowledgments
- [Udacity](https://www.udacity.com/) for providing the course content.
- [Figure Eight](https://www.figure-eight.com/) for the dataset.

## Additional sources
Discussions on testing a weights
https://medium.com/analytics-vidhya/optimize-hyperparameters-with-gridsearch-d351b0fd339d

Lemmatization
https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
