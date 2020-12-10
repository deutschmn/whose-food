# %%

import pandas as pd
import regex as re
import matplotlib.pyplot as plt
from pathlib import Path
import os
from shutil import copy


# %%

def load_data(chat_file):
    with open(chat_file) as f:
        content = f.readlines()

    str = "\n".join(content)
    m = re.compile("\[([^\]]*)\] ([^:]*):.*< pièce jointe : ([^ ]*) >").findall(
        str)

    df = pd.DataFrame(m, columns=["date", "from", "photo"])

    df["date"] = pd.to_datetime(df["date"])

    # some data validation: filter out some unneeded entries

    # - first entry not needed, just some icon change
    df = df.drop(0)

    # - some entries are videos, we only look at the JPGs
    df = df[df["photo"].apply(lambda x: x.endswith(".jpg"))]

    return df


# %%

def show_plots(df):
    df.groupby(['from']).count()["photo"].plot(kind='bar')
    plt.title("Top reporters")
    plt.tight_layout()
    plt.show()

    df.groupby(df['date'].dt.month).count()["photo"].plot(kind='bar')
    plt.title("Photos per month")
    plt.xlabel("month")
    plt.show()

    df.groupby(df['date'].dt.hour).count()["photo"].plot(kind='bar')
    plt.title("Photos per hour")
    plt.xlabel("hour of the day")
    plt.show()

    hours = df.groupby(['from', df['date'].dt.hour]).count()["photo"]
    for k in df['from'].unique():
        plt.plot(hours[k].index, hours[k].values, label=k)
    plt.legend()
    plt.xlabel("hour of the day")
    plt.show()


# %%

data_dir = "data"
all_photos_dir = os.path.join(data_dir, "photos")
split_photos_dir = os.path.join(data_dir, "from")

df = load_data(os.path.join(data_dir, "_chat.txt"))

show_plots(df)

# %%
df.groupby('from')
for reporter, group in df.groupby('from'):
    target_dir = os.path.join(split_photos_dir, reporter)
    Path(target_dir).mkdir(exist_ok=True, parents=True)

    for photo in group["photo"]:
        src = os.path.join(all_photos_dir, photo)
        dst = os.path.join(target_dir, photo)
        copy(src, dst)
