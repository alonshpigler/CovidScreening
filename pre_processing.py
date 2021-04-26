import pandas as pd

CHANNELS = (1,2,3,4,5)
LABEL_FIELD = []

def split_by_channel(filename, task_channel):
    """
    This function is responsible for splitting five channels into four channels as train and the remaining channel to test
    filename: file path to the cell table from a single plate
    task_channel: the current channel that we aim to predict
    Notably: In order to avoid leakage we drop all 'correlation features
    return: separated dataframes x_features and y_df.
            x_features: contains all available features excluding the features related to 'task_channel' we aim to predict
            y_df: contains all available features related to 'task_channel' only
    """

    # Data preparation
    df = load_plate_csv(filename)
    df.dropna(inplace=True)

    general_cols, corr_cols, dict_channel_cols = list_columns(df)

    not_curr_channel_cols = [col for channel in CHANNELS if channel != task_channel
                             for col in dict_channel_cols[channel]]
    cols = general_cols + not_curr_channel_cols

    x_features_df = df[cols]

    y_df = df[dict_channel_cols[task_channel]]

    return x_features_df, y_df


def load_plate_csv(csv_file):
    """
    This function read a plate's csv file and give an indexed pandas dataframe
    """
    df: pd.DataFrame = pd.read_csv(csv_file)
    df = df.set_index(
        ['Plate', LABEL_FIELD, 'Metadata_broad_sample', 'Image_Metadata_Well', 'ImageNumber', 'ObjectNumber'])
    return df



def list_columns(df):
    """
    This function split the plate's columns to different lists
    general columns, correlation columns and a dictionary channel -> channel columns
    """
    general_cols = [f for f in df.columns if all(c not in f for c in CHANNELS)]
    corr_cols = [f for f in df.columns if 'Correlation' in f]

    # Split columns by channel
    dict_channel_cols = {}
    for channel in CHANNELS:
        dict_channel_cols[channel] = [col for col in df.columns if channel in col and col not in corr_cols]

    return general_cols, corr_cols, dict_channel_cols


def split_train_test(path, csv_files, test_plate, task_channel):
    """
    This function prepare dataframes for the models
    test contain the data from one channel and train contains only mock from other plates
    """
    # Prepare test samples
    df_test_x, df_test_y = split_by_channel(path + test_plate, task_channel)

    df_test_mock_x = df_test_x[df_test_x.index.isin(['mock'], 1)]
    df_test_treated_x = df_test_x[df_test_x.index.isin(['treated'], 1)]
    df_test_mock_y = df_test_y.loc[df_test_mock_x.index]
    df_test_treated_y = df_test_y.loc[df_test_treated_x.index]

    # Prepare train samples - only mock
    list_x_df = []
    list_y_df = []
    for train_plate in tqdm(csv_files):
        if train_plate != test_plate:
            curr_x, curr_y = split_by_channel(path + train_plate, task_channel)
            curr_x = curr_x[curr_x.index.isin(['mock'], 1)]
            curr_y = curr_y.loc[curr_x.index]

            list_x_df.append(curr_x)
            list_y_df.append(curr_y)

    df_train_x = pd.concat(list_x_df)
    df_train_y = pd.concat(list_y_df)

    return df_test_mock_x, df_test_mock_y, df_test_treated_x, df_test_treated_y, df_train_x, df_train_y