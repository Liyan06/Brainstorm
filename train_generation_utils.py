import pandas as pd
from typing import List, Dict
import logging
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn


logger = logging.getLogger(__name__)


def shift_tokens_left(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    prev_output_tokens[:, :-1] = input_ids[:, 1:]
    prev_output_tokens[:, -1] = pad_token_id
    return prev_output_tokens


def prepare_input_output(args, df_dir) -> List[Dict]:
    df = pd.read_csv(df_dir)

    if args.do_cl_training_with_margin:
        examples = []
        for _, row in df.iterrows():
            examples.append(
                {'input': row.input, 'output': row.output, 'neg_input': row.neg_input})
    elif args.do_simcse_training:
        examples = []
        examples = []
        for _, row in df.iterrows():
            examples.append(
                {'input': row.input, 'output': row.output, 'neg_input': row.neg_input, 'neg_output': row.neg_output})
    else:
        input_name = args.input_name

        examples = []
        for _, row in df.iterrows():
            if input_name == 'neg_input':
                examples.append(
                    {'input': row[input_name], 'output': row.neg_output})
            else:
                examples.append(
                    {'input': row[input_name], 'output': row.output})

    return examples


def get_examples(args, data_dir) -> List[Dict]:
    return prepare_input_output(args, df_dir=data_dir)


class InputFeatures(object):
    def __init__(self):
        self.have_neg = 0
        self.labels = None
        self.neg_input_ids = None
        self.neg_attention = None

    def setup_common_features(self, input_ids, decoder_ids, attention, decoder_attention):
        self.input_ids = input_ids
        self.decoder_ids = decoder_ids
        self.attention = attention
        self.decoder_attention = decoder_attention


def convert_examples_to_features(args, examples, tokenizer, input_max_len, output_max_len) -> List[InputFeatures]:
    features = []

    pad_id = tokenizer.pad_token_id
    assert pad_id is not None

    for example in examples:

        input_text = example['input']
        output_text = example['output']

        inputs = tokenizer.batch_encode_plus(
            [input_text], max_length=input_max_len, truncation=True, padding='max_length', return_tensors="pt")

        outputs = tokenizer.batch_encode_plus(
            [output_text], max_length=output_max_len, truncation=True, padding='max_length', return_tensors="pt")

        input_ids, input_attention_mask = inputs['input_ids'].to(
            torch.long), inputs['attention_mask'].to(torch.long)
        decoder_ids, decoder_attention_mask = outputs['input_ids'].to(
            torch.long), outputs['attention_mask'].to(torch.long)

        input_features = InputFeatures()

        if args.do_cl_training_with_margin:

            neg_input_text = example['neg_input']
            neg_inputs = tokenizer.batch_encode_plus(
                [neg_input_text], max_length=input_max_len, truncation=True, padding='max_length', return_tensors="pt")
            neg_input_ids, neg_input_attention_mask = neg_inputs[
                'input_ids'].to(torch.long), neg_inputs['attention_mask'].to(torch.long)

            if args.model_type == 'bart':
                input_features.setup_common_features(
                    input_ids=input_ids,
                    attention=input_attention_mask,
                    decoder_attention=decoder_attention_mask,
                    decoder_ids=decoder_ids,
                )
                input_features.neg_input_ids = neg_input_ids
                input_features.neg_attention = neg_input_attention_mask

                features.append(input_features)
        elif args.do_simcse_training:

            neg_input_text = example['neg_input']
            neg_inputs = tokenizer.batch_encode_plus(
                [neg_input_text], max_length=input_max_len, truncation=True, padding='max_length', return_tensors="pt")
            neg_input_ids, neg_input_attention_mask = neg_inputs[
                'input_ids'].to(torch.long), neg_inputs['attention_mask'].to(torch.long)

            neg_output_text = example['neg_output']
            neg_outputs = tokenizer.batch_encode_plus(
                [neg_output_text], max_length=output_max_len, truncation=True, padding='max_length', return_tensors="pt")
            neg_decoder_ids, neg_decoder_attention_mask = neg_outputs[
                'input_ids'].to(torch.long), neg_outputs['attention_mask'].to(torch.long)

            if args.model_type == 'bart':
                input_features.setup_common_features(
                    input_ids=input_ids,
                    attention=input_attention_mask,
                    decoder_attention=decoder_attention_mask,
                    decoder_ids=decoder_ids,
                )

                input_features.neg_input_ids = neg_input_ids
                input_features.neg_attention = neg_input_attention_mask
                input_features.neg_decoder_ids = neg_decoder_ids
                input_features.neg_decoder_attention = neg_decoder_attention_mask

                features.append(input_features)
        else:

            neg = 0

            if args.model_type == 'bart':
                input_features.setup_common_features(
                    input_ids=input_ids,
                    attention=input_attention_mask,
                    decoder_attention=decoder_attention_mask,
                    decoder_ids=decoder_ids,
                )
                input_features.have_neg = neg

                features.append(input_features)


    assert len(features) == len(examples)
    return features


def load_examples(args, tokenizer, category):
    if category == 'train':
        data_dir = '/'.join(args.train_df_dir.split('/')[:-1])
    elif category == 'val':
        data_dir = '/'.join(args.val_df_dir.split('/')[:-1])
    elif category == 'test':
        data_dir = '/'.join(args.brain_test_df_dir.split('/')[:-1])
    else:
        raise AssertionError("invalid category.")

    logger.info("Creating features from dataset file at %s: %s", data_dir,
                'dev' if category == 'val' else 'train' if category == 'train' else 'test')
    print(
        f"Creating features for: {'dev' if category == 'val' else 'train' if category == 'train' else 'test'}.")

    examples = get_examples(args, args.val_df_dir) if category == 'val' \
        else get_examples(args, args.train_df_dir) if category == 'train' \
        else get_examples(args, args.test_df_dir)

    features = convert_examples_to_features(
        args,
        examples,
        tokenizer,
        input_max_len=args.max_input_length,
        output_max_len=args.max_output_length,
    )

    input_ids = torch.stack([f.input_ids for f in features]).squeeze(1)
    input_attention_mask = torch.stack(
        [f.attention for f in features]).squeeze(1)

    if args.do_cl_training_with_margin:
        neg_input_ids = torch.stack(
            [f.neg_input_ids for f in features]).squeeze(1)
        neg_input_attention_mask = torch.stack(
            [f.neg_attention for f in features]).squeeze(1)
    elif args.do_simcse_training:
        neg_input_ids = torch.stack(
            [f.neg_input_ids for f in features]).squeeze(1)
        neg_input_attention_mask = torch.stack(
            [f.neg_attention for f in features]).squeeze(1)
        neg_decoder_ids = torch.stack(
            [f.neg_decoder_ids for f in features]).squeeze(1)
        neg_decoder_attention_mask = torch.stack(
            [f.neg_decoder_attention for f in features]).squeeze(1)
    else:
        have_neg = torch.tensor(
            [f.have_neg for f in features], dtype=torch.long)

    decoder_ids = torch.stack(
        [f.decoder_ids for f in features]).squeeze(1)
    decoder_attention_mask = torch.stack(
        [f.decoder_attention for f in features]).squeeze(1)

    if args.do_cl_training_with_margin:
        return TensorDataset(input_ids, input_attention_mask, decoder_ids, decoder_attention_mask, neg_input_ids, neg_input_attention_mask)
    elif args.do_simcse_training:
        return TensorDataset(input_ids, input_attention_mask, decoder_ids, decoder_attention_mask, neg_input_ids, neg_input_attention_mask, neg_decoder_ids, neg_decoder_attention_mask)
    else:
        return TensorDataset(input_ids, input_attention_mask, decoder_ids, decoder_attention_mask, have_neg)

# https://github.com/yixinl7/brio
class label_smoothing(nn.Module):
    def __init__(self, ignore_index, epsilon=0.1):
        super(label_smoothing, self).__init__()
        self.ignore_idx = ignore_index
        self.epsilon = epsilon

    def forward(self, input, target):
        input = torch.log_softmax(input, dim=2)
        k = input.size(2)
        target_prob = torch.ones_like(input).type_as(
            input) * self.epsilon * 1 / k
        mask = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(
            target.size(0), target.size(1), -1).type_as(target)
        mask = torch.eq(mask, target.unsqueeze(-1).expand(-1, -1, k))
        target_prob.masked_fill_(
            mask, 1 - self.epsilon + (self.epsilon * 1 / k))
        loss = - torch.mul(target_prob, input)
        loss = loss.sum(2)
        # mask ignore_idx
        mask = (target != self.ignore_idx).type_as(input)
        loss = (torch.mul(loss, mask).sum() / mask.sum()).mean()
        return loss
