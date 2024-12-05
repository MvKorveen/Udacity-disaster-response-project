import sys
import pandas as pd
from sqlalchemy import create_engine
from langdetect import detect
import langcodes

def load_data(messages_filepath, categories_filepath):
        # load messages dataset
        df_messages = pd.read_csv(messages_filepath)

        def detect_language(text):
            try:
                # Check if the text is empty or None
                if not text:
                    return 'Undetected'
                lang_code = detect(text)
                # Get the full language name using langcodes
                return langcodes.Language.get(lang_code).language_name()
            except Exception:
                return 'Undetected'
            
        # Apply the function to the 'original' column
        df_messages['language'] = df_messages['original'].apply(detect_language)

        # load categories dataset
        df_categories = pd.read_csv(categories_filepath)

        # merge datasets
        df=df_messages.merge(df_categories, on='id', how='left', suffixes=('_messages', '_categories'))

        return df


def clean_data(df):

    # create a dataframe of the 36 individual category columns
    df_message_categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    first_row = df_message_categories.iloc[:1,:]

    # create an empty list for the extracted column names
    category_colnames = []

    # Loop through columns in the DataFrame
    for col in first_row.columns:
        # Extract value for col, removing last two caracters
        temp_new_col = first_row[col].iloc[0][:-2]

        # Append the extracted value to the new_col list
        category_colnames.append(temp_new_col)
    
    # rename the columns of `categories`
    df_message_categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.

    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    for column in df_message_categories:
        # set each value to be the last character of the string
        df_message_categories[column] = df_message_categories[column].str[-1]
    
        # convert column from string to numeric
        df_message_categories[column] = df_message_categories[column].astype('int')

    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, df_message_categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///{0}'.format(database_filename))
    df.to_sql('{0}'.format(database_filename), engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()