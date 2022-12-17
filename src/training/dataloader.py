from os.path import abspath, splitext
from typing import List, Optional, Union

from datasets import load_dataset


def load(
    tokenizer,
    seq_len: int,
    train_data_path: Union[str, List[str]],
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    shuffle_seed: Optional[int] = None,
):
    def process_one_file(data):  # 방(파일) 하나만 가져옴
        data = {k: v[0] for k, v in data.items()}  # 배치 처리
        input_text = []

        # room_name = data["filename"]
        persona_features = "당신의 성격:\n" + "\n".join(data["personaFeatures"])

        for sess in data["session"]:
            if not sess["dialog"]:
                continue

            prev_hist_summary = [
                f"{k} 이전 대화 요약: " + "\n".join(sess["prevAggregatedpersonaSummary"][k])
                for k in ["speaker1", "speaker2"]
            ]

            time = sess["prevTimeInfo"]["timeNum"] + sess["prevTimeInfo"]["timeUnit"]
            time = (time + " 후, ") if time else ""

            last_speaker = sess["dialog"][0]["speaker"]
            dialogue = [""]
            summaries = [""]
            for uttr in sess["dialog"]:
                if uttr["speaker"] != last_speaker:
                    dialogue[-1] = last_speaker + ":" + dialogue[-1]
                    summaries[-1] = last_speaker + ":" + summaries[-1]

                    dialogue.append("")
                    summaries.append("")

                    last_speaker = uttr["speaker"]

                dialogue[-1] += " " + uttr["utterance"]
                summaries[-1] += " " + uttr["summary"]

            dialogue[-1] = last_speaker + ":" + dialogue[-1]
            summaries[-1] = last_speaker + ":" + summaries[-1]

            accumulator = "대화 요약:"
            prev_summary = ""
            prev_dialog = ""
            for di, su in zip(dialogue, summaries):
                if su.startswith("speaker1"):
                    text = "\n\n".join(
                        [
                            persona_features,
                            "\n".join(prev_hist_summary),
                            time + accumulator,
                            "다음 내용에 답하세요: "
                            + (
                                "" if not prev_dialog else prev_dialog.split(": ", 1)[1]
                            ),
                            "답변: " + di.split(": ", 1)[1],
                        ]
                    )
                    input_text.append(text)
                accumulator += (
                    ("\n" + prev_summary) if prev_hist_summary else prev_summary
                )
                prev_summary = su
                prev_dialog = di

        return {"input_text": input_text}

    def tokenize(data):
        tokenize = tokenizer(
            data["input_text"],
            max_length=seq_len + 1,
            padding="max_length",
            truncation=True,
        )

        return {
            "input_ids": tokenize.input_ids[:-1],
            "attention_mask": tokenize.attention_mask[:-1],
            "labels": tokenize.input_ids[1:],
        }

    train_data_path = abspath(train_data_path)
    is_eval = False
    _, extention = splitext(train_data_path)

    datafiles = {"train": train_data_path}

    if eval_data_path is not None:
        datafiles["test"] = abspath(eval_data_path)
        is_eval = True

    if train_test_split is not None:
        train_test_split = int(train_test_split * 100)
        train_test_split = {
            "train": f"train[:{train_test_split}%]",
            "test": f"train[{train_test_split}%:]",
        }
        is_eval = True

    data = load_dataset(
        extention.replace(".", ""),
        data_files=datafiles,
        split=train_test_split,
    )

    data = data.map(
        process_one_file,
        batched=True,
        batch_size=1,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    data = data.map(
        tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    return data["train"], (data["test"] if is_eval else None)
