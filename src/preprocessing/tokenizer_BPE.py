import hydra
import json
import pandas as pd


def load_jsonl(input_path) -> list:
    """
    for jsonl load

    Args:
        input_path (str): path for json file load

    Returns:
        list: json to list
    """

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    return data


@hydra.main(config_name="../config.yaml")
def main(cfg):
    """
    main executive unit
    """

    df = pd.DataFrame(load_jsonl(cfg.PATH.train_data))
    print("df : {}".format(df.head()))


if __name__ == "__main__":
    main()
