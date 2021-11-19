import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads raw data.
    """

    # load messages dataset
    messages=pd.read_csv(messages_filepath)
    # load categories dataset
    categories=pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id', how='left')

    return df

def clean_data(df):
    """
    This function cleans data from the raw data.
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # extract a list of new column names for categories.
    category_colnames = categories.iloc[0].apply(lambda x: x[:-2]).values
    # rename the columns of `categories`
    categories.columns = category_colnames
    # change each value into 0/1 with dtype of int64
    for column in categories:
        categories[column] = categories[column].apply(lambda x:int(x[-1]))
    # convert the values which are 2 in related column to 0
    categories.loc[categories['related']==2,'related']=0
    # drop the original categories column from `df`
    df=df.drop('categories',axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # Drop the original column
    df.drop('original',axis=1,inplace=True)

    return df

def save_data(df, database_filename):
    """
    This function saves data.
    """

    # create a database and save the table
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterRecord', engine, if_exists = 'replace', index=False)

def main():
    """
    This is the main function load all the functions above.
    """

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
