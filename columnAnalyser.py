import operator
import pprint as pp
from datetime import datetime as dtime

import numpy as np
import pandas as pd

SUM_OF_PROB = "SUM_OF_PROB"
ONLY_PROB = "ONLY_PROB"
ENUM = "CATEGORICAL"

EQUALS = "="
COMMA = ","
MAX_COL_DEFAULT = -10000
MIN_COL_DEFAULT = 10000
DEFAULT_AGE_RANGE = "ALL"
DEFAULT_COLUMN_VALUE = "DEFAULT_VALUE"
DEFAULT_DURATION = -1
DEFAULT_DURATION_DAYS = "None"
DEFAULT_MAX_BID_VALUE = -1
ITERATION_NUMBER = 12
TRAIN_TO_DATA_PROPORTION = 0.7
FORMAT = '%m/%d/%y %H:%M'
SKIP_VALUES_WITH_ZERO_IMPRESSIONS = True
CATEGORICAL_TREATMENT = ENUM


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


def process_categorical(column_name, output_matrix, given_categories):
    column_array = df[column_name]
    process_categorical_post_row_extract(column_array, column_name, given_categories, output_matrix)


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


def process_categorical_summation_of_probabilities(column_name, output_matrix, given_categories):
    column_array = df[column_name]
    process_categorical_summation_of_probabilities_post_column_extract(column_array, column_name, given_categories,
                                                                       output_matrix)


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


def process_categorical_probabilities(column_name, output_matrix, given_categories):
    column_array = df[column_name]
    process_categorical_features_as_probabilities_post_extract(column_array, column_name, given_categories,
                                                               output_matrix)


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


def process_categorical_as_designed_post_extract(processed_duration, column_name, output_matrix, given_categories):
    if CATEGORICAL_TREATMENT == SUM_OF_PROB:
        process_categorical_summation_of_probabilities_post_column_extract \
            (processed_duration, column_name, given_categories, output_matrix)
    elif CATEGORICAL_TREATMENT == ONLY_PROB:
        process_categorical_features_as_probabilities_post_extract \
            (processed_duration, column_name, given_categories, output_matrix)
    else:
        process_categorical_post_row_extract \
            (processed_duration, column_name, given_categories, output_matrix)


def process_categorical_as_designed(column_name, output_matrix, given_categories):
    if CATEGORICAL_TREATMENT == SUM_OF_PROB:
        process_categorical_summation_of_probabilities(column_name, output_matrix, given_categories)
    elif CATEGORICAL_TREATMENT == ONLY_PROB:
        process_categorical_probabilities(column_name, output_matrix, given_categories)
    else:
        process_categorical(column_name, output_matrix, given_categories)


#######################################################CODE BEGINS#####################################################
df = pd.read_csv("input/CocaColaCompany_CocaCola_TweetEngagements.csv")
output_matrix = pd.DataFrame()

process_non_categorical(output_matrix, "Max Bid", DEFAULT_MAX_BID_VALUE)

age_categories = ['AGE_18_TO_34', 'AGE_18_TO_49', 'AGE_13_TO_24', 'AGE_OVER_25', 'AGE_13_TO_49', 'AGE_21_TO_49',
                  'AGE_21_TO_34', 'AGE_OVER_18', 'AGE_13_TO_34']
country_categories = ['Brazil', 'Australia', 'Italy', 'Mexico', 'South Africa', 'France', 'Chile', 'Japan', 'Ecuador']

process_categorical_as_designed('Ad Delivery Status', output_matrix, ["Delivering"])
process_categorical_as_designed('Billing Event', output_matrix, ["ENGAGEMENT"])
process_categorical_as_designed('Optimization Goal', output_matrix, ["ENGAGEMENT"])
process_categorical_as_designed('Automatically Set Bid', output_matrix, ["Yes"])
process_categorical_as_designed('Bid Type', output_matrix, ["CPA"])
process_categorical_as_designed('Budget Pacing', output_matrix, ["Standard"])
process_categorical_as_designed('Age Range', output_matrix, age_categories)
process_categorical_as_designed("Gender", output_matrix, ["Male", "Female"])
# process_categorical_as_designed('Countries', output_matrix, country_categories)

###########################CUSTOM COLUMNS############################
add_duration_column(output_matrix)

process_frequency_capping(output_matrix)

#########################CONVERION###################################
CONVERSION = "conversion"
CLICKS = 'Twitter Post Link Clicks'
IMPRESSIONS = 'Twitter Posts Impressions'

clicks = df[CLICKS]
impressions = df[IMPRESSIONS]

list_to_skip = []

processed_ad_delivery = []
for index, click in enumerate(clicks):
    clickFloat = float(click)
    impression = float(impressions[index])
    if clickFloat == 0 or impression == 0:
        processed_ad_delivery.append(0)
        if SKIP_VALUES_WITH_ZERO_IMPRESSIONS:
            list_to_skip.append(index)
    else:
        conversion = clickFloat / impression
        processed_ad_delivery.append(conversion)
        # if conversion < 0.005:
        #     processedAdDelivery.append(conversion) #Manual outlier removal
        # else:
        #     processedAdDelivery.append(0)
        #     listToSkip.append(index)

output_matrix.loc[:, CONVERSION] = pd.Series(processed_ad_delivery)

msf_filter = [True for i in range(len(output_matrix))]
for term in list_to_skip:
    msf_filter[term] = False
msf_filter = np.array(msf_filter)
output_matrix = output_matrix[msf_filter]

msk = np.random.rand(len(output_matrix)) < TRAIN_TO_DATA_PROPORTION

train, test = output_matrix[msk], output_matrix[~msk]

# output_matrix.to_csv("featuresUsedIn" + str(ITERATION_NUMBER) + "thIteration.csv")
train.to_csv("featureDataUsedIn" + str(ITERATION_NUMBER) + "thIteration_TRAIN.csv")
test.to_csv("featureDataUsedIn" + str(ITERATION_NUMBER) + "thIteration_TEST.csv")

feature_file = open("ListOfFeaturesIn" + str(ITERATION_NUMBER) + "thIteration.txt", 'w')
for item in list(train.columns.values):
    feature_file.write("%s\n" % item)

# output_matrix.to_csv("outputFeatures.csv", header=False)
train.to_csv("outputFeaturesTrain.csv", header=False)
test.to_csv("outputFeaturesTest.csv", header=False)

pp.pprint(output_matrix)
