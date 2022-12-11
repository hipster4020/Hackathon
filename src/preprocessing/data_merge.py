import os
import config
import json
import pandas as pd
from tqdm import tqdm


def main():
    """
    json to dataframe(train/validation/test)

    :param url:
    :return:
    """

    # Data Load
    train_data = []
    val_data = []
    test_data = []
    data = [[] for _ in config.data]

    for k, d in tqdm(enumerate(config.data)):

        for i in range(3):
            path = os.path.join(config.pwd, "session_{}".format(i + 2), str(d))
            entries = os.listdir(path)
            for e in entries:
                with open(os.path.join(path, str(e))) as f:
                    for j in json.load(f)["sessionInfo"][0]["dialog"]:
                        data[k].append(j)
    train_data, val_data, test_data = data

    # json to dataframe
    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)
    df_test = pd.DataFrame(test_data)

    # data save
    s_list = ["train", "val", "test"]
    for s in s_list:
        path = config.pwd + s + ".json"

        temp_dict = [
            {
                "speaker": row["speaker"].strip(),
                "personaID": row["personaID"].strip(),
                "utterance": row["utterance"].strip(),
                "summary": row["summary"].strip(),
                "terminate": row["terminate"].strip(),
            }
            for _, row in eval("df_" + s).iterrows()
        ]
        with open(path, "w", encoding="utf-8") as f:
            for line in temp_dict:
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + "\n")
        print("Saved {} : {} records".format(eval("df_" + s), len(eval("df_" + s))))


if __name__ == "__main__":
    main()
