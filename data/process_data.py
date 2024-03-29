import sys

import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load two CSV files from messages_filepath and categories_filepath,
    and create a pandas DataFrame object for each CSV file, finally
    merge the two DataFrame objects on rows and return the merged 
    DataFrame object.

    input:
        messages_filepath: The path of the messages CSV file.
        categories_filepath: The path of the categories CSV file.

    return:
        df: a pandas DataFrame object that contains the content of 
            the merge of the input files.
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories)

    return df

def clean_data(df):
    '''
    Clean the input pandas DataFrame in the following steps:
        1. Split categories column into 36 individual category columns;
        2. Convert category values to just numbers 0 or 1;
        3. Remove any duplicate rows.

    input:
        df: A pandas DataFrame object.

    return：
        df: The cleaned pandas DataFrame object.
    '''
    # Split categories into separate category columns.
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')

    # Remove duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    '''
    Save a pandas DataFrame into a sqlite database.

    input:
        df: The pandas DataFrame object to be saved.
        database_filename: the name of the sqlite database file.
    '''
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql('ResponseCategory', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filename = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        save_data(df, database_filename)
        
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