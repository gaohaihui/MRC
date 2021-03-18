import json
import args
import torch
import pickle
import torch.nn as nn
from tqdm import tqdm

import predict_data
from tokenization import BertTokenizer
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForQuestionAnswering, BertConfig
from MRC.dataset.util import read_squad_examples

def find_best_answer(start_probs, end_probs):
  best_start, best_end, max_prob = -1, -1, 0

  prob_start, best_start = torch.max(start_probs, 1)
  prob_end, best_end = torch.max(end_probs, 1)
  num = 0
  while True:
      if num > 3:
          break
      if best_end >= best_start:
          break
      else:
          start_probs[0][best_start], end_probs[0][best_end] = 0.0, 0.0
          prob_start, best_start = torch.max(start_probs, 1)
          prob_end, best_end = torch.max(end_probs, 1)
      num += 1
  max_prob = prob_start * prob_end

  if best_start <= best_end:
      return (best_start, best_end), max_prob
  else:
      return (best_end, best_start), max_prob

def evaluate(model, eval_path, result_file):

    eval_examples = read_squad_examples(eval_path)

    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained(r'D:\self\Graduation\data\bert', do_lower_case=True)
        model.eval()
        pred_answers, ref_answers = [], []

        for step, example in enumerate(tqdm(eval_examples)):
            start_probs, end_probs = [], []
            question_text = example['question_text']
            (input_ids, input_mask, segment_ids) = predict_data.predict_data(question_text, example['doc_tokens'], tokenizer, args.max_seq_length, args.max_query_length)

            start_prob, end_prob = model(input_ids, segment_ids, attention_mask=input_mask)     # !!!!!!!!!!

            best_span, docs_index = find_best_answer(start_prob, end_prob)
            para = "p" + example['question_text'] + "。" + example['doc_tokens']
            best_answer = ''.join(para[best_span[0]: best_span[1]+1])
            pred_answers.append({'question_id': example['id'],
                                 'question':example['question_text'],
                                 'question_type': example['question_type'],
                                 'answers': [best_answer],
                                 'entity_answers': [[]],
                                 'yesno_answers': []})
            if 'answer' in example:
                ref_answers.append({'question_id': example['id'],
                                    'question_type': example['question_type'],
                                    'answers': example['answer'],
                                    'entity_answers': [[]],
                                    'yesno_answers': []})
        with open(result_file, 'w', encoding='utf-8') as fout:
            for pred_answer in pred_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
        with open("../metric/ref.json", 'w', encoding='utf-8') as fout:
            for pred_answer in ref_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

def eval_all(eval_path):
   
    output_model_file = "../model_dir/best_model"
    output_config_file = "../model_dir/bert_config.json"  
    
    config = BertConfig(output_config_file)
    model = BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(output_model_file)) #, map_location='cpu'))
    evaluate(model.cpu(),eval_path, result_file="predicts.json")

eval_path = r"D:\self\Graduation\MRC\data\dev.json"
eval_all(eval_path)
