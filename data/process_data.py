import sys
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load messages_file and categories file into dataframe for processing
    INPUT: file path for messages_file and categories_file
    OUTPUT: pandas dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.concat([messages,categories],axis = 1)
    return df


def clean_data(df):
    '''
    Clean dataframe to modify data structure to assist future analyzation
    INPUT: pandas dataframe
    OUTPUT: pandas dataframe after cleaning
    '''
    # create a dataframe of the 36 individual category columns
    df_cat = df.categories.str.split(';',expand = True)
    # select the first row of the categories dataframe
    row = df_cat.values.tolist()[0]
    new = []
    
    for char in row[0:37]:
        new.append(char.split('-')[0])
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = new
    
    # rename the columns of `categories`
    df_cat.columns = category_colnames
    
    # simplify contents in columns
    for char in new:
        df_cat[char] = df_cat[char].apply(lambda x:x.split('-')).apply(lambda y:y[1])
        df_cat.drop(df_cat[df_cat[char]>'1'].index,inplace = True)
    df_cat_new = df_cat
    
    # concatenate the original dataframe with the new `categories` dataframe
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, df_cat_new],axis =1 )
    #pd.concat([df_cat['id'],df_cat.categories.str.split(';',expand = True)],axis = 1)
    df = df.loc[:,~df.columns.duplicated()]
    # drop duplicates
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    return df
    
def save_data(df, database_filepath):
    '''
    Load processed data into database
    INPUT: pandas dataframe, database_filepath
    OUTPUT: SQL database file table name 'messages_disaster'
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('messages_disaster', engine, index=False,if_exists = 'replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
      
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
          
        df = load_data(messages_filepath, categories_filepath)
        print(df.shape)
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