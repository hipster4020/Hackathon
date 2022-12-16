from os.path import abspath, splitext
from typing import List, Optional, Union

from datasets import load_dataset, logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import default_data_collator

logging.set_verbosity(logging.ERROR)


def load(
    tokenizer,
    seq_len: int,
    train_data_path: Union[str, List[str]],
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1,
    shuffle_seed: Optional[int] = None,
):
    def _grouping(data):
        """grouping for datasets map

        Args:
            data (class): datasets.arrow_dataset.Batch

        Returns:
            json: for datasets.arrow_dataset.Batch
        """
        # {'filename': ['K2-06611-CL30516-CP40880-04-08-S2.json'], 'personaFeatures': [['나는 남자이다.', '나는 30대이다.', '나는 국내 주식에 관심이 많다', '나는 매우 활발하다', '나는 칼국수를 안 먹는다']], 'session': [[{'prevTimeInfo': {'timeNum': '', 'timeUnit': ''}, 'dialog': [{'speaker': 'speaker1', 'utterance': '안녕하세요 저는 30대 남성이에요', 'summary': '나는 30대 남성이다.'}, {'speaker': 'speaker2', 'utterance': '안녕하세요. 저는 40대 남성입니다. 반갑습니다.', 'summary': '나는 40대 남성이다'}, {'speaker': 'speaker1', 'utterance': '반갑습니다 요즘 저는 국내 주식에 관심이 많아요', 'summary': '나는 국내 주식에 관심이 많다.'}, {'speaker': 'speaker2', 'utterance': '주식에 관심이 많으시군요. 저는 주식하다가 크게 날려먹은걸 배우자가 처가에 고자질해서 배우자의 부모님과 사이가 어색한 사이가 되버렸습니다.', 'summary': '나는 배우자의 부모님과 어색한 사이가 되었다.'}, {'speaker': 'speaker1', 'utterance': '아이고 저런 ㅜ 그런 경우도 있군요 저도 돈 잃은 적도 있지만 제테크로 해야할 거 같아서요', 'summary': ''}, {'speaker': 'speaker2', 'utterance': '뭐 어쩔수 없죠. 이미 지나간일이니까요. 실례하지만 직업이 어떻게 되세요? 저는 도배공 일을 하고 있습니다.', 'summary': '나는 도배공이다'}, {'speaker': 'speaker1', 'utterance': '아 저는 국내 주식을 전업투자로 하고 있어요 도배공일이라니 아주 멋진 직업이네요', 'summary': '나는 국내 주식 전업투자자다.'}, {'speaker': 'speaker2', 'utterance': '저는 좋은 직업이라 생각하는데 도배공이 노가다라는 선입견 때문에 배우자의 부모님과 어색한 사이랍니다. 제 직업을 마음에 안들어하세요.', 'summary': ''}, {'speaker': 'speaker1', 'utterance': '저는 성격이 매우 활발해서 몸을 쓰는 일을 좋아해요 도배공 일도 그래서 너무 멋져보여요', 'summary': '나는 성격이 매우 활발하다.'}, {'speaker': 'speaker2', 'utterance': '저도 몸쓰는일 좋아합니다. 취미도 드럼 연주에요. 드럼 연주를 자주 하곤 합니다.', 'summary': '나는 드럼 연주를 자주 한다.'}, {'speaker': 'speaker1', 'utterance': '우와 멋지네요 혹시 좋아하는 음식은 뭘까 궁금해요 저는 칼국수는 안 먹고 다른건 다 잘 먹어요', 'summary': '나는 칼국수를 싫어한다.'}, {'speaker': 'speaker2', 'utterance': '칼국수를 싫어하나봐요? 저는 싱싱한 회를 좋아해요.', 'summary': '나는 싱싱한 회를 좋아한다'}, {'speaker': 'speaker1', 'utterance': '아 저도 회 좋아해요 없어서 못먹죠 ㅋ', 'summary': '나는 회를 좋아한다.'}, {'speaker': 'speaker2', 'utterance': '음식 얘기 하다보니 배가 고파졌네요. 식사하러 가야겠어요. 다음에 또 뵐께요.', 'summary': ''}], 'prevAggregatedpersonaSummary': {'speaker1': [], 'speaker2': []}}, {'prevTimeInfo': {'timeNum': '40', 'timeUnit': '시간'}, 'dialog': [{'speaker': 'speaker1', 'utterance': '40시간 만이네요  잘 지내셨나요?', 'summary': ''}, {'speaker': 'speaker2', 'utterance': '오랜만이네요. 저는 도배공 급여 정산중이었어요.', 'summary': ''}, {'speaker': 'speaker1', 'utterance': '아 급여 정산 중이셨군요  저도 급여 받았어요 ', 'summary': '나는 급여를 받았다.'}, {'speaker': 'speaker2', 'utterance': '많이 받으셨어요? 급여가 팍팍 들어와야 일하는데 기분도 더 좋을텐데 말이죠.', 'summary': ''}, {'speaker': 'speaker1', 'utterance': '얼마 안되요 작고 소중한 수준이죠 ', 'summary': '나는 급여가 적다.'}, {'speaker': 'speaker2', 'utterance': '최저 임금이 오르면서 그래도 급여가 몇년사이에 참 많이 올랐어요.', 'summary': '몇 년 사이 급여가 많이 올랐다'}, {'speaker': 'speaker1', 'utterance': '아 맞아요 급여가 오른 경우가 많죠 ', 'summary': ''}, {'speaker': 'speaker2', 'utterance': '우리 도배공들도 급여가 참 많이 올랐어요. 일꾼들 일당들 오르는만큼 시공비도 올라가죠.', 'summary': '나는 급여가 많이 올랐다'}, {'speaker': 'speaker1', 'utterance': '아 그렇군요 부럽습니다 좋으시겠어요', 'summary': ''}, {'speaker': 'speaker2', 'utterance': '전업 투자자이신데 급여를 받나요? 어떤 급여를 받나요?', 'summary': ''}, {'speaker': 'speaker1', 'utterance': '전업투자로는 배당금 받고 사외이사로 급여 받고있어요', 'summary': '나는 배당금을 받고 사외이사 급여를 받는다.'}, {'speaker': 'speaker2', 'utterance': '전업투자도 다양한 급여를 받을수가 있군요. 전혀 몰랐어요.', 'summary': ''}, {'speaker': 'speaker1', 'utterance': '아 그런편이죠 하지만 급여만으로 살기에는 빠듯해요', 'summary': ''}, {'speaker': 'speaker2', 'utterance': '무슨일이든 점점 살기 어려워지는것 같아요. 저는 이제 다시 일을 하러 가봐야겠어요. 다음에 또 뵐께요.', 'summary': ''}], 'prevAggregatedpersonaSummary': {'speaker1': ['나는 30대 남성이다.', '나는 국내 주식에 관심이 많다.', '나는 국내 주식 전업투자자다.', '나는 성격이 매우 활발하다.', '나는 칼국수를 싫어한다.', '나는 회를 좋아한다.'], 'speaker2': ['나는 40대 남성이다', '나는 배우자의 부모님과 어색한 사이가 되었다.', '나는 도배공이다', '나는 드럼Exception ignored in: <generator object tqdm.__iter__ at 0x7fb11c586350>
        result = {}

        # personaFeatures
        personaFeatures = tokenizer(
            sum(data["personaFeatures"], []),
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        result["persona_input_ids"] = personaFeatures["input_ids"]
        result["persona_attention_mask"] = personaFeatures["attention_mask"]

        # prevent_session_input_ids, prevent_session_input_ids, prevent_session_attention_mask, main_input_ids, main_attention_mask, decoder_input_ids, decoder_attention_mask, labels

        # input_ids = []
        # last_input_ids = []
        # last_room_no = data["room_no"][0]
        # # merge room_no
        # for ii, wr, ti in zip(encoded, data["speaker"], data["room_no"]):
        #     if len(last_input_ids + ii) <= seq_len + 1 and last_room_no == ti:
        #         last_input_ids += ii
        #     else:
        #         input_ids.append(last_input_ids)
        #         last_input_ids = ii
        #         last_room_no = ti
        # data = {"input_ids": input_ids}

        return data

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
        _grouping,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )
