import operator
import pprint as pp
from datetime import datetime as dtime

import numpy as np
import pandas as pd

SUM_OF_PROB = "SUM_OF_PROB"
ONLY_PROB = "ONLY_PROB"
ENUM = "CATEGORICAL"
EQUIDISTANT = "EQUIDISTANT"

EQUALS = "="
COMMA = ","
MAX_COL_DEFAULT = -10000
MIN_COL_DEFAULT = 10000
DEFAULT_AGE_RANGE = "ALL"
DEFAULT_COLUMN_VALUE = "DEFAULT_VALUE"
DEFAULT_MAX_BID_VALUE = 0.9999
DEFAULT_DURATION = DEFAULT_MAX_BID_VALUE
DEFAULT_DURATION_DAYS = "None"
ITERATION_NUMBER = "COKE_LRG_5"
DATA_PORTION = 1
TRAIN_TO_TOTAL_DATA = 0.7
FORMAT = '%m/%d/%y %H:%M'
SKIP_VALUES_WITH_ZERO_IMPRESSIONS = True
CATEGORICAL_TREATMENT = ONLY_PROB


def process_non_categorical(output_matrix, column_name, default_value):
    column_values = df[column_name]
    process_non_categorical_post_column_extraction(column_values, column_name, default_value, output_matrix)


def process_non_categorical_post_column_extraction(column_values, column_name, default_value, output_matrix):
    min_col = MIN_COL_DEFAULT
    max_col = MAX_COL_DEFAULT
    processed_ad_delivery = []
    for i in column_values:
        if pd.isnull(i):
            processed_ad_delivery.append(default_value)
        else:
            floatValue = float(i)

            min_col = floatValue if floatValue < min_col else min_col
            max_col = floatValue if floatValue > max_col else max_col
            processed_ad_delivery.append(floatValue)
    range_value = max_col - min_col
    for index, processedValue in enumerate(processed_ad_delivery):
        if processedValue != DEFAULT_MAX_BID_VALUE:
            processed_ad_delivery[index] = (processedValue - min_col) / range_value
    output_matrix.loc[:, column_name] = pd.Series(processed_ad_delivery)


def add_duration_column(output):
    TIME_COLUMN = "Duration"
    AD_SET_START_TIME = 'Ad Set Time Start'
    CAMPAIGN_START_TIME = 'Campaign Time Start'
    AD_SET_STOP_TIME = 'Ad Set Time Stop'
    CAMPAIGN_STOP_TIME = 'Campaign Time Stop'

    ad_set_start_time_column = df[AD_SET_START_TIME]
    pi_start_time_column = df[CAMPAIGN_START_TIME]
    ad_set_stop_time_column = df[AD_SET_STOP_TIME]
    pi_stop_time_column = df[CAMPAIGN_STOP_TIME]
    processed_ad_delivery = []
    min_duration = 100000
    max_duration = -100000
    for index, time in enumerate(ad_set_stop_time_column):
        end_time = None
        if not pd.isnull(time):
            end_time = dtime.strptime(time, FORMAT)
        elif not pd.isnull(pi_stop_time_column[index]):
            end_time = dtime.strptime(pi_stop_time_column[index], FORMAT)

        start_time = None
        if not pd.isnull(ad_set_start_time_column[index]):
            start_time = dtime.strptime(ad_set_start_time_column[index], FORMAT)
        elif not pd.isnull(pi_start_time_column[index]):
            start_time = dtime.strptime(pi_start_time_column[index], FORMAT)

        if end_time == None or start_time == None:
            processed_ad_delivery.append(DEFAULT_DURATION)
        else:
            duration = (end_time - start_time).total_seconds()
            min_duration = duration if duration < min_duration else min_duration
            max_duration = duration if duration > max_duration else max_duration
            processed_ad_delivery.append(duration)
    duration_range = max_duration - min_duration
    for index, processedValue in enumerate(processed_ad_delivery):
        if processedValue == DEFAULT_DURATION:
            continue
        processed_ad_delivery[index] = (processedValue - min_duration) / duration_range
    output.loc[:, TIME_COLUMN] = pd.Series(processed_ad_delivery)


def provide_list_of_probabilities(column_dict, processed_column):
    total = len(processed_column)
    probabilities_ad_delivery = []
    for i, value in enumerate(processed_column):
        number_of_entries = column_dict.get(value, 0)
        probabilities_ad_delivery.append(float(number_of_entries) / total)
    return probabilities_ad_delivery


def increment_value_for_key_in_dict(delivery_dict, key):
    number = delivery_dict.get(key, 0)
    delivery_dict[key] = number + 1


def process_categorical_post_row_extract(column_array, column_name, given_categories, output_matrix):
    processed_column_array = []
    default_category = len(given_categories) + 1
    for column_entry in column_array:
        column_value = default_category
        for index, category in enumerate(given_categories):
            if column_entry == category:
                column_value = index + 1
                break
        processed_column_array.append(column_value)
    output_matrix.loc[:, column_name] = pd.Series(processed_column_array)


def process_categorical_summation_of_probabilities_post_column_extract(column_array, column_name, given_categories,
                                                                       output_matrix):
    column_dict = dict()
    processed_column_array = []
    default_category = len(given_categories) + 1
    for column_entry in column_array:
        column_value = default_category
        for index, category in enumerate(given_categories):
            if column_entry == category:
                column_value = index + 1
                break
        processed_column_array.append(column_value)
        increment_value_for_key_in_dict(column_dict, column_value)
    probabilities_dict = dict()
    summation = 0
    sorted_x = sorted(column_dict.items(), key=operator.itemgetter(1))
    total = len(processed_column_array)
    for tuple in sorted_x:
        summation += float(tuple[1]) / total
        probabilities_dict[tuple[0]] = summation
    probabilities_ad_delivery = []
    for column_value in processed_column_array:
        probabilities_ad_delivery.append(probabilities_dict.get(column_value))
    output_matrix.loc[:, column_name] = pd.Series(probabilities_ad_delivery)


def process_categorical_features_as_probabilities_post_extract(column_array, column_name, given_categories,
                                                               output_matrix):
    column_dict = dict()
    processed_column_array = []
    default_category = len(given_categories) + 1
    for column_entry in column_array:
        column_value = default_category
        if not pd.isnull(column_entry):
            for category_index, category in enumerate(given_categories):
                if column_entry == category:
                    column_value = category_index + 1
                    break
        processed_column_array.append(column_value)
        increment_value_for_key_in_dict(column_dict, column_value)
    probabilities_dict = dict()
    total = len(processed_column_array)
    for tuple in column_dict.items():
        probabilities_dict[tuple[0]] = float(tuple[1]) / total

    probabilities_ad_delivery = []
    for column_value in processed_column_array:
        probabilities_ad_delivery.append(probabilities_dict.get(column_value))
    output_matrix.loc[:, column_name] = pd.Series(probabilities_ad_delivery)


def process_categorical_features_equidistant_post_extract(column_array, column_name, given_categories,
                                                          output_matrix):
    column_dict = dict()
    processed_column_array = []
    default_category = len(given_categories)
    for column_entry in column_array:
        column_value = default_category
        for index, category in enumerate(given_categories):
            if column_entry == category:
                column_value = index
                break
        processed_column_array.append(column_value)
        column_dict[column_value] = 1
    len_categories = float(len(column_dict.keys()))
    probabilities_ad_delivery = []
    for column_value in processed_column_array:
        probabilities_ad_delivery.append(float(column_value) / len_categories)
    output_matrix.loc[:, column_name] = pd.Series(probabilities_ad_delivery)


def process_frequency_capping(output_matrix):
    column_array = df["Frequency Control"]
    duration_dict = dict({'1': 'Day', '7': 'Week', '30': 'Month'})
    processed_frequency_control = []
    processed_duration = []
    for column_entry in column_array:
        if pd.isnull(column_entry):
            processed_frequency_control.append(np.NaN)
            processed_duration.append(DEFAULT_DURATION_DAYS)
        else:
            parts = column_entry.strip().split(COMMA)
            frequency_cap = parts[1].strip().split(EQUALS)[1]
            duration_value = parts[0].strip().split(EQUALS)[1]
            column_value = DEFAULT_DURATION_DAYS
            for tuple in duration_dict.items():
                if duration_value == tuple[0]:
                    column_value = tuple[1]
                    break
            processed_frequency_control.append(frequency_cap)
            processed_duration.append(column_value)

    process_non_categorical_post_column_extraction(processed_frequency_control, "Frequency Cap",
                                                   DEFAULT_MAX_BID_VALUE, output_matrix)
    process_categorical_as_designed_post_extract(processed_duration, "Duration Days", output_matrix,
                                                 ['Day', 'Week', 'Month', 'None'])


def distribute_data_based_on_portion(output_matrix, list_to_skip):
    # noinspection PyUnusedLocal
    clear_zero_values_mask = [True for i in range(len(output_matrix))]
    for term in list_to_skip:
        clear_zero_values_mask[term] = False
    clear_zero_values_mask = np.array(clear_zero_values_mask)
    output_matrix = output_matrix[clear_zero_values_mask]

    partial_data_msk = np.random.rand(len(output_matrix)) < DATA_PORTION
    partial_output = output_matrix[partial_data_msk]

    train_to_data_msk = np.random.rand(len(partial_output)) < TRAIN_TO_TOTAL_DATA
    train, test = partial_output[train_to_data_msk], partial_output[~train_to_data_msk]
    return output_matrix, train, test


def get_labels_for_column(column_name):
    column = df[column_name]
    column_dict = dict()
    for label in column:
        if pd.isnull(label):
            column_dict["NULL"] = 1
            continue
        column_dict[label] = 1
    return column_dict.keys()


def twitter_conversion(output_matrix, converted_column_name):
    CONVERSION_RATE = "conversion"
    IMPRESSIONS = 'Twitter Posts Impressions'
    impressions = df[IMPRESSIONS]

    CONVERTED = converted_column_name
    converted_column = df[CONVERTED]

    list_to_skip = []
    processed_ad_delivery = []
    for index, converted_value in enumerate(converted_column):
        converted_float = float(converted_value)
        impression = float(impressions[index])
        if impression == 0:
            processed_ad_delivery.append(0)
            if SKIP_VALUES_WITH_ZERO_IMPRESSIONS:
                list_to_skip.append(index)
        else:
            conversion_rate = converted_float / impression
            if conversion_rate < 100.01: #all cases
                processed_ad_delivery.append(conversion_rate) #Manual outlier removal
            else:
                processed_ad_delivery.append(0)
                list_to_skip.append(index)
    output_matrix.loc[:, CONVERSION_RATE] = pd.Series(processed_ad_delivery)
    return list_to_skip


def process_categorical_as_designed_post_extract(column_entries, column_name, output_matrix, given_categories):
    if CATEGORICAL_TREATMENT == SUM_OF_PROB:
        process_categorical_summation_of_probabilities_post_column_extract \
            (column_entries, column_name, given_categories, output_matrix)
    elif CATEGORICAL_TREATMENT == ONLY_PROB:
        process_categorical_features_as_probabilities_post_extract \
            (column_entries, column_name, given_categories, output_matrix)
    elif CATEGORICAL_TREATMENT == EQUIDISTANT:
        process_categorical_features_equidistant_post_extract \
            (column_entries, column_name, given_categories, output_matrix)
    else:
        process_categorical_post_row_extract \
            (column_entries, column_name, given_categories, output_matrix)


def process_categorical_as_designed(column_name, output_matrix, given_categories):
    process_categorical_as_designed_post_extract(df[column_name], column_name, output_matrix, given_categories)


#######################################################CODE BEGINS#####################################################
# df = pd.read_excel("input/twitter_mz_active_completed_deleted_across_partner.xlsx", sheetname="Sheet1")
# df = pd.read_csv("input/twitter_mz_active_completed_deleted_across_partner.csv")
df = pd.read_csv("input/CokeAllDataPromotedTweets.csv")
output_matrix = pd.DataFrame()

process_non_categorical(output_matrix, "Max Bid", DEFAULT_MAX_BID_VALUE)

process_categorical_as_designed('Ad Delivery Status', output_matrix, ["Delivering"])
process_categorical_as_designed('Billing Event', output_matrix, get_labels_for_column("Billing Event"))
process_categorical_as_designed('Optimization Goal', output_matrix, get_labels_for_column('Optimization Goal'))
process_categorical_as_designed('Automatically Set Bid', output_matrix, get_labels_for_column("Automatically Set Bid"))
process_categorical_as_designed('Bid Type', output_matrix, get_labels_for_column("Bid Type"))
process_categorical_as_designed('Budget Pacing', output_matrix, get_labels_for_column("Budget Pacing"))
#process_categorical_as_designed('Age range', output_matrix, get_labels_for_column('Age range'))

process_categorical_as_designed("Gender", output_matrix, get_labels_for_column("Gender"))
# process_categorical_as_designed('Countries', output_matrix, country_categories)

# process_categorical_as_designed("Match Relevant Topics", output_matrix, get_labels_for_column("Match Relevant Topics"))
# process_categorical_as_designed("Positive Sentiment", output_matrix, get_labels_for_column("Positive Sentiment"))
# process_categorical_as_designed("User Operating System", output_matrix, get_labels_for_column("User Operating System"))
# process_categorical_as_designed("User OS Version", output_matrix, get_labels_for_column("User OS Version"))

###########################CUSTOM COLUMNS############################
add_duration_column(output_matrix)

process_frequency_capping(output_matrix)

#########################CONVERSION###################################
list_to_skip = twitter_conversion(output_matrix, 'Twitter Post Link Clicks')

########################################################################
output_matrix, train, test = distribute_data_based_on_portion(output_matrix, list_to_skip)

# output_matrix.to_csv("featuresUsedIn" + str(ITERATION_NUMBER) + "thIteration.csv")
train.to_csv("featureDataUsedIn" + str(ITERATION_NUMBER) + "thIteration_TRAIN_" + CATEGORICAL_TREATMENT + ".csv")
test.to_csv("featureDataUsedIn" + str(ITERATION_NUMBER) + "thIteration_TEST_" + CATEGORICAL_TREATMENT + ".csv")

feature_file = open("ListOfFeaturesIn" + str(ITERATION_NUMBER) + "thIteration_" + CATEGORICAL_TREATMENT + ".txt", 'w')
for item in list(train.columns.values):
    feature_file.write("%s\n" % item)

# output_matrix.to_csv("outputFeatures.csv", header=False)
train.to_csv("outputFeaturesTrain.csv", header=False)
test.to_csv("outputFeaturesTest.csv", header=False)

pp.pprint(output_matrix)
