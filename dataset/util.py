import json
from tqdm import tqdm

def load_json(path):
  with open(path, 'r', encoding="utf-8") as f:
    s1 = json.load(f)
    return s1

class InputFeatures(object):
  """
  Input feature for BERT.
  """
  def __init__(self, input_ids, input_mask, segment_ids, start_position,
               end_position):
    """
    Args:
      input_id(int): Id of character.
      label_id(str): Id of label for classify.
      input_mask(int): Mask flag, 0 or 1.
    """
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position


def read_squad_examples(data_path):
  examples = []
  data = load_json(data_path)
  data = data["data"][0]["paragraphs"]
  for line in data:
    example = {}
    context = line["context"]
    qas = line["qas"][0]
    question = qas["question"]
    id = qas["id"]
    answer = qas["answers"][0]["text"]
    start_position = qas["answers"][0]["answer_start"]
    example["start_position"] = start_position
    example["end_position"] = start_position + len(answer)
    example["question_text"] = question
    example["question_type"] = "DESCRIBE"
    example["answer"] = answer
    example["doc_tokens"] = context
    example["id"] = id
    examples.append(example)
  print("len(examples):", len(examples))
  return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
  features = []

  for example in tqdm(examples):
    query_tokens = list(example['question_text'])
    question_type = example['question_type']

    doc_tokens = example['doc_tokens']
    doc_tokens = doc_tokens.replace(u"“", u"\"")
    doc_tokens = doc_tokens.replace(u"”", u"\"")
    start_position = example['start_position']
    end_position = example['end_position']

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)
    start_position = start_position + 1
    end_position = end_position + 1

    for token in query_tokens:
      tokens.append(token)
      segment_ids.append(0)
      start_position = start_position + 1
      end_position = end_position + 1

    tokens.append("[SEP]")
    segment_ids.append(0)
    start_position = start_position + 1
    end_position = end_position + 1

    for i in doc_tokens:
      tokens.append(i)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    if end_position >= max_seq_length:
      continue

    if len(tokens) > max_seq_length:
      tokens[max_seq_length - 1] = "[SEP]"
      input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])  ## !!! SEP
      segment_ids = segment_ids[:max_seq_length]
    else:
      input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)
    assert len(input_ids) == len(segment_ids)
    extra_len = (max_seq_length - len(input_ids))
    input_ids += extra_len * [0]
    input_mask += extra_len * [0]
    segment_ids += extra_len * [0]
    features.append(InputFeatures(input_ids=input_ids,input_mask= input_mask,segment_ids=segment_ids,
       start_position=start_position,
       end_position=end_position))
  print("len(features):", len(features))
  return features