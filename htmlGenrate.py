#coding:utf-8

import os
import pandas as pd


def convert_to_title(title_index):

    if title_index == 0:
        title = "First Party Collection/Use"
    elif title_index == 1:
        title = "Third Party Sharing/Collection"
    elif title_index == 2:
        title = "User Choice/Control"
    elif title_index == 3:
        title = "User Access, Edit, & Deletion"
    elif title_index == 4:
        title = "Data Retention"
    elif title_index == 5:
        title = "Data Security"
    elif title_index == 6:
        title = "Policy Change"
    elif title_index == 7:
        title = "Do Not Track"
    elif title_index == 8:
        title = "International & Specific Audiences"
    elif title_index == 9:
        title = "Other"
    else:
        title = "something else"
    return "<h2> " + title + "</h2>"


file_list = os.listdir("./predict_result")


for file_name in file_list:
    body_segments = []
    title_index = -1
    policy = os.path.join('./predict_result', file_name)
    messages = pd.read_csv(policy, header=None)
    for index, row in messages.iterrows():
        if row[0] > title_index:
            title_index = row[0]
            body_segments.append(convert_to_title(title_index))
        body_segments.append("<p> " + row[1] + "</p>")
    policy_name = file_name.replace(".txt.tsv.csv", "")
    f = open("./html_result/" + policy_name + ".html", "wb")
    f.writelines(body_segments)

