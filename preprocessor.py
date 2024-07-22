import re
import pandas as pd


def preprocess(data):
    # Define regex patterns for both 24-hour and 12-hour time formats
    split_formats_12hr = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM][\s\u200B]*-\s'
    split_formats_24hr = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{2}:\d{2}\s-\s'

    # Split messages and dates for both formats
    messages_12hr = re.split(split_formats_12hr, data)
    dates_12hr = re.findall(split_formats_12hr, data)
    messages_24hr = re.split(split_formats_24hr, data)
    dates_24hr = re.findall(split_formats_24hr, data)

    # Combine messages and dates, preferring 12-hour format if it matches
    if len(messages_12hr) > 1:
        messages = messages_12hr[1:]
        dates = dates_12hr
        datetime_format_1 = '%d/%m/%Y, %I:%M %p - '
        datetime_format_2 = '%d/%m/%y, %I:%M %p - '
    else:
        messages = messages_24hr[1:]
        dates = dates_24hr
        datetime_format_1 = '%d/%m/%Y, %H:%M - '
        datetime_format_2 = '%d/%m/%y, %H:%M - '

    # Create DataFrame
    df = pd.DataFrame({'User_Message': messages, 'Date': dates})

    # Attempt to parse dates with multiple formats
    try:
        df['Date'] = pd.to_datetime(df['Date'], format=datetime_format_1, dayfirst=True)
    except ValueError:
        df['Date'] = pd.to_datetime(df['Date'], format=datetime_format_2, dayfirst=True)

    # Separate Users and Messages
    users = []
    messages = []
    for message in df['User_Message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notifications')
            messages.append(entry[0])

    df['User'] = users
    df['Message'] = messages
    df.drop(columns=['User_Message'], inplace=True)

    # Extract additional date and time components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month_name()
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['Minute'] = df['Date'].dt.minute
    df['Month_num'] = df['Date'].dt.month
    df['_Date_'] = df['Date'].dt.date
    df["Day_name"] = df['Date'].dt.day_name()

    # Create Period column
    period = []
    for hour in df['Hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['Period'] = period

    return df
