from pprint import pprint
import pandas as pd
from sys import exit
import json

NA_LIST = ['', 'NA']

def convert(df):

    df['Condition'] = df['Condition'].astype(str)

    def convert_with_padding(x):
        try:
            return str(int(x)).zfill(3)
        except ValueError:
            return x  # or whatever you want to do with non-numeric values

    df['Condition'] = df['Condition'].apply(convert_with_padding)

    sentence_cols = df.filter(regex='^Sentence_').columns
    df[sentence_cols] = df[sentence_cols].fillna('')
    df[sentence_cols] = df[sentence_cols].astype(str)

    # Get all column names that start with 'Sentence_'
    sentence_cols = [col for col in df.columns if col.startswith('Sentence_')]

    # Extract the i and j indices from the column names
    indices = [list(map(int, col.split('_')[1:])) for col in sentence_cols]

    # Derive the outer and inner list lengths
    outer_list_len = max(idx[0] for idx in indices)
    inner_list_len = max(idx[1] for idx in indices)

    def create_nested_sentences(row, outer_list_len, inner_list_len):
        nested_sentences = []
        for i in range(outer_list_len):
            inner_list = []
            for j in range(inner_list_len):
                value = row[f'Sentence_{i+1}_{j+1}']
                if value not in NA_LIST:
                    inner_list.append(value)
            if inner_list:  # Check if inner list is not empty
                nested_sentences.append(inner_list)
        return nested_sentences

    df['sentences'] = df.apply(
        create_nested_sentences,
        args=(outer_list_len, inner_list_len),
        axis=1
    )

    # drop the sentence columns
    df = df.drop(columns=sentence_cols)

    # Create separate dataframes for producers and receivers
    common_cols = ["Exp", "Round", "Item", "Condition", "Item.Nr", "sentences"]
    roles_cols = ["Producer", "Interpreter"]
    df_producers = df[roles_cols + common_cols].copy()
    df_receivers = df[roles_cols + common_cols].copy()

    # Add role column
    df_producers["role"] = "P"
    df_receivers["role"] = "I"

    # Rename id columns
    df_producers.rename(
        columns={"Producer": "id", "Interpreter": "partner_id"}, inplace=True
    )
    df_receivers.rename(
        columns={"Interpreter": "id", "Producer": "partner_id"}, inplace=True
    )

    # Concatenate the dataframes
    df_long = pd.concat([df_producers, df_receivers])

    # Reorder the columns
    df_long = df_long[[*common_cols, "role", "id", "partner_id"]]

    # drop rows with role P and id == 0
    df_long = df_long.loc[~((df_long['role'] == 'P') & (df_long['id'] == 0))]

    sort_by = ['Exp', 'Round', 'id']
    df_long.sort_values(by=sort_by, inplace=True)
    df_long['group_enumeration'] = df_long.groupby(['Exp', 'id']).cumcount() + 1
    df_long.sentences = df_long.sentences.apply(json.dumps)
    return df_long


if __name__ == "__main__":
    file_path = './test.csv'
    df = pd.read_csv(file_path, dtype={"Condition": str})
    df_long = convert(df)
    df_long.sentences = df_long.sentences.apply(json.dumps)
    print(df_long.head())
    df_long.to_csv('./mocklong.csv', index=False)
