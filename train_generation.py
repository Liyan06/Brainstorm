from model_arch import *
from utils import common_args
import json
import os
import random
import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import BartTokenizer, get_linear_schedule_with_warmup
import train_generation_utils
from torch.optim import AdamW


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bart": (
        BartConfig,
        InterpretBart,
        BartTokenizer
    )
}


def seed_everything(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def save_checkpoints(args, output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, eval_dataset, model, tokenizer, gen_dataset=None) -> Dict:

    eval_output_dir = args.eval_output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model, device_ids=args.gpu_device_ids)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Eval Batch size = %d", args.eval_batch_size)
    eval_loss_sentence = 0.0
    nb_eval_steps = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():

            if args.do_cl_training_with_margin:
                if args.model_type == 'bart':
                    input_ids, attention, decoder_ids, decoder_attention, neg_input_ids, neg_input_attention_mask = batch[
                        0], batch[1], batch[2], batch[3], batch[4], batch[5]
                    inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                              'decoder_attention_mask': decoder_attention, 'neg_input_ids': neg_input_ids,
                              'neg_attention_mask': neg_input_attention_mask}
            elif args.do_simcse_training:
                input_ids, attention, decoder_ids, decoder_attention, neg_input_ids, neg_input_attention_mask, neg_decoder_ids, neg_decoder_attention_mask = batch[
                    0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]
                inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                            'decoder_attention_mask': decoder_attention,
                            'neg_input_ids': neg_input_ids,
                            'neg_attention_mask': neg_input_attention_mask, 'neg_decoder_ids': neg_decoder_ids,
                            'neg_decoder_attention_mask': neg_decoder_attention_mask}
            else:
                input_ids, attention, decoder_ids, decoder_attention, have_neg = batch[
                    0], batch[1], batch[2], batch[3], batch[4]
                inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                            'decoder_attention_mask': decoder_attention, 'have_neg': have_neg}


            outputs = model(**inputs)
            tmp_eval_loss_sentence = outputs[0]

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel training
                tmp_eval_loss_sentence = tmp_eval_loss_sentence.mean()
            eval_loss_sentence += tmp_eval_loss_sentence.item()

            nb_eval_steps += 1

    result = {'loss': eval_loss_sentence/nb_eval_steps}
    print(result)

    if args.generate:

        df = pd.DataFrame(columns=["source", "reference", "output"])
        with open(os.path.join(eval_output_dir, 'test_out.txt' if args.do_test else 'dev_out.txt'), 'w') as f_out:
            k = 0

            gen_sampler = SequentialSampler(gen_dataset)
            gen_dataloader = DataLoader(
                gen_dataset, sampler=gen_sampler, batch_size=args.eval_batch_size)

            for batch in tqdm(gen_dataloader):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                input_ids, input_attention_mask, decoder_ids = batch[0], batch[1], batch[2]

                for j in range(input_ids.shape[0]):

                    gold = tokenizer.decode(
                        decoder_ids[j], skip_special_tokens=True)
                    input = tokenizer.decode(
                        input_ids[j], skip_special_tokens=False)

                    if args.diverse_bs:
                        generation = model.generate(
                            input_ids[j].unsqueeze(0), attention_mask=input_attention_mask[j].unsqueeze(0),
                            num_beams=args.num_beams, length_penalty=args.length_penalty, no_repeat_ngram_size=3,
                            max_length=args.max_output_length, min_length=args.min_output_length,
                            decoder_start_token_id=tokenizer.bos_token_id, use_cache=False,
                            num_return_sequences=args.num_return_sequences, top_p=args.top_p, top_k=args.top_k,
                            num_beam_groups=args.num_beams, diversity_penalty=1.0,
                            do_sample=args.do_sample, early_stopping=True
                        )
                    else:
                        generation = model.generate(
                            input_ids[j].unsqueeze(0), attention_mask=input_attention_mask[j].unsqueeze(0),
                            num_beams=args.num_beams, length_penalty=args.length_penalty, no_repeat_ngram_size=3,
                            max_length=args.max_output_length, min_length=args.min_output_length,
                            decoder_start_token_id=tokenizer.bos_token_id, use_cache=False,
                            num_return_sequences=args.num_return_sequences, top_p=args.top_p, top_k=args.top_k,
                            do_sample=args.do_sample, early_stopping=True
                        )    
                    

                    f_out.write('input:\n' + input.replace("<pad>", "") + '\n')
                    f_out.write('reference:\n' + gold + '\n')

                    top_gen = tokenizer.decode(
                        generation[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    for rank in range(args.num_return_sequences):
                        gen = tokenizer.decode(
                            generation[rank], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        gen = gen[:gen.find(".")+1]

                        print(gen.strip())
                        if len(gen) == 0:
                            continue

                        f_out.write(
                            f'generation top {rank+1}: ' + gen.strip() + '\n')

                    print()
                    f_out.write('\n')

                    top_gen = top_gen[:top_gen.find(".")+1]
                    df.loc[len(df.index)] = [input, gold, top_gen.strip()]

                k += 1

        df.to_csv(args.ref_sum_name, index=False)

    output_eval_file = os.path.join(
        eval_output_dir, "test_results.txt" if args.do_test else 'eval_results.txt')
    with open(output_eval_file, "a") as writer:
        logger.info(
            "***** Test results *****" if args.do_test else "***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write('\n')

    return result


def train(args, train_dataset, eval_dataset, model, tokenizer, gen_dataset) -> Tuple[int, float]:
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model, device_ids=args.gpu_device_ids)

    if args.max_steps > 0:
        num_train_steps = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        num_train_steps = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_steps
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Train Batch size = %d", args.train_batch_size)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    # logger.info("  args.local_rank = %d", args.local_rank)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_train_steps)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, tr_loss_sent, logging_loss_sent = 0.0, 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    seed_everything(args)

    torch.cuda.empty_cache()
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            batch = tuple(t.to(args.device) for t in batch)

            if args.do_cl_training_with_margin:
                input_ids, attention, decoder_ids, decoder_attention, neg_input_ids, neg_input_attention_mask = batch[
                    0], batch[1], batch[2], batch[3], batch[4], batch[5]
                inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                            'decoder_attention_mask': decoder_attention, 'neg_input_ids': neg_input_ids,
                            'neg_attention_mask': neg_input_attention_mask}
            elif args.do_simcse_training:
                input_ids, attention, decoder_ids, decoder_attention, neg_input_ids, neg_input_attention_mask, neg_decoder_ids, neg_decoder_attention_mask = batch[
                    0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]
                inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                            'decoder_attention_mask': decoder_attention,
                            'neg_input_ids': neg_input_ids,
                            'neg_attention_mask': neg_input_attention_mask, 'neg_decoder_ids': neg_decoder_ids,
                            'neg_decoder_attention_mask': neg_decoder_attention_mask}
            else:
                input_ids, attention, decoder_ids, decoder_attention, have_neg = batch[
                    0], batch[1], batch[2], batch[3], batch[4]
                inputs = {'input_ids': input_ids, 'attention_mask': attention, 'decoder_input_ids': decoder_ids,
                            'decoder_attention_mask': decoder_attention, 'have_neg': have_neg}

            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_loss_sent += loss.item()
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    logs = {}
                    learning_rate_scalar = scheduler.get_last_lr()
                    logs["learning_rate"] = learning_rate_scalar
                    loss_scalar_sent = (
                        tr_loss_sent - logging_loss_sent) / args.save_steps
                    logs["loss_sent"] = loss_scalar_sent
                    logging_loss_sent = tr_loss_sent

                    print('\n', json.dumps({**logs, **{"step": global_step}}))
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                    # Evaluation
                    evaluate(args, eval_dataset, model,
                             tokenizer)
                    save_checkpoints(
                        args, args.params_output_dir, model, tokenizer)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        save_checkpoints(args, args.params_output_dir, model, tokenizer)
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    evaluate(args, eval_dataset, model, tokenizer, gen_dataset)

    return global_step, tr_loss / global_step


if __name__ == "__main__":

    parser = common_args()
    args = parser.parse_args()

    device = torch.device("cuda", args.gpu_device)

    args.device = device
    seed_everything(args)

    if (os.path.exists(args.params_output_dir)
            and os.listdir(args.params_output_dir)
            and args.do_train
            and not args.overwrite_output_dir
            ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overwrite.".format(
                args.params_output_dir
            )
        )

    if (os.path.exists(args.params_output_dir)
            and os.listdir(args.params_output_dir)
            and args.do_train
            and args.overwrite_output_dir
            ):
        logger.info("Overwrite previous parameters.")
        print("Overwrite previous parameters.")

    if not os.path.exists(args.params_output_dir):
        os.makedirs(args.params_output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.params_output_dir, 'model.log')
    )

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.input_dir is not None:
        print(f'loading model from {args.input_dir}')
        tokenizer = tokenizer_class.from_pretrained(args.input_dir)
        model = model_class.from_pretrained(args.input_dir)
    else:
        print('loading pre-trained model')
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config)

    model.to(args.device)

    if args.do_cl_training_with_margin:
        model.use_cl_training_with_ref()

    if args.do_simcse_training:
        model.use_simcse_training()

    if args.do_train or args.do_eval:
        eval_dataset = train_generation_utils.load_examples(
            args, tokenizer, category='val')
        if args.generate:
            gen_dataset = train_generation_utils.load_examples(
                args, tokenizer, category='val')
        else:
            gen_dataset = None

    if args.do_train:
        logger.info("Training parameters %s", args)
        train_dataset = train_generation_utils.load_examples(
            args, tokenizer, category='train')
        global_step, tr_loss = train(
            args, train_dataset, eval_dataset, model, tokenizer, gen_dataset)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

    if args.do_eval:
        print("Start Evaluation.")
        evaluate(args, eval_dataset, model, tokenizer,
                 gen_dataset)
        logger.info("Evaluation parameters %s", args)
