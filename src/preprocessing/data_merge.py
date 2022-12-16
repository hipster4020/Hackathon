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
            for e, ee in enumerate(entries):
                with open(os.path.join(path, str(ee))) as f:
                    rawfile = json.load(f)
                # personaFeatures, prevTimeInfo, dialog:speaker, utterance, summary, prevAggregatedpersonaSummary
                temp = {}
                temp["filename"] = rawfile["FileInfo"]["filename"]
                temp["personaFeatures"] = rawfile["personaInfo"]["clInfo"][
                    "personaFeatures"
                ]
                temp["session"] = [
                    {
                        "prevTimeInfo": session["prevTimeInfo"],
                        "dialog": [
                            {kk: row[kk] for kk in ["speaker", "utterance", "summary"]}
                            for row in session["dialog"]
                        ],
                        "prevAggregatedpersonaSummary": session[
                            "prevAggregatedpersonaSummary"
                        ],
                    }
                    for session in rawfile["sessionInfo"]
                ]
                data[k].append(temp)
    train_data, val_data, test_data = data

    # data save
    s_list = ["train", "val", "test"]
    for s in s_list:
        path = os.path.join(config.pwd + s + ".json")

        # temp_dict = [
        #     {
        #         "speaker": row["speaker"].strip(),
        #         "personaID": row["personaID"].strip(),
        #         "utterance": row["utterance"].strip(),
        #         "summary": row["summary"].strip(),
        #         "terminate": row["terminate"].strip(),
        #         "conv_no": row["conv_no"],
        #     }
        #     for _, row in eval("df_" + s).iterrows()
        # ]

        with open(path, "w", encoding="utf-8") as f:
            for line in eval(f"{s}_data"):
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + "\n")
        print("Saved {} : {} records".format(f"{s}_data", len(f"{s}_data")))


if __name__ == "__main__":
    main()
