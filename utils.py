import argparse
import numpy as np
import pandas as pd

from pythonrouge.pythonrouge import Pythonrouge


def common_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-medcat_vocab", default="./MedCAT/vocab.dat")
    parser.add_argument("-medcat_model", default="./MedCAT/cdb_mimic_md_21-April-2021.dat")

    parser.add_argument("--train_df_dir", default="data/brain_mri/brain_train_df_0117.csv")
    parser.add_argument("--val_df_dir", default="data/brain_mri/brain_val_df_0117.csv")
    parser.add_argument("--input_name", default="input")

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        help="Model type selected in the list: [bart, gpt2, t5]",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="facebook/bart-base",
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--params_output_dir",
        default='./params/',
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--eval_output_dir",
        default='./eval_output/',
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_input_length",
        default=-1,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_output_length",
        default=128,
        type=int,
        help="The maximum total output sequence length after tokenization.",
    )
    parser.add_argument(
        "--min_output_length",
        default=10,
        type=int,
        help="The minimum total output sequence length after tokenization.",
    )
    parser.add_argument(
        "--ref_sum_name",
        default=None,
        type=str,
        help="The directory to store (reference, summary) dataframe + file name.",
    )
    parser.add_argument(
        "--external_encoding",
        default=None,
        type=str,
        help="path that stores the encoding of external knowledge.",
    )
 
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")

    parser.add_argument("--diverse_bs", action="store_true")
    parser.add_argument("--do_cl_training_with_margin", action="store_true", help="use contrastive learning for current proposed method I: max( 0, P(y|x,~i)-P(y|x, i)+epsilon)")
    parser.add_argument("--do_simcse_training", action="store_true", help="use simcse training")

    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--n_gpu", default=1, type=int, help="Number of Gpu to use.", )
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size training.", )
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int, help="Batch size evaluation.", )
    parser.add_argument("--gpu_device", default=0, type=int, help="gpu device")
    parser.add_argument("--gpu_device_ids", nargs="+", type=int, default=[0, 1, 2, 3], help="gpu device")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="No. steps before backward pass.",)
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs", )
    parser.add_argument("--max_steps", default=-1, type=int, help="If>0: no. train steps. Overrides num_train_epochs.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory", )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--generate", action="store_true", help="Generate summaries for dev set", )
    parser.add_argument("--seed", type=int, default=2021, help="random seed for initialization")
    parser.add_argument("--num_beams", type=int, default=6, help="beamsize")
    parser.add_argument("--length_penalty", type=int, default=2, help="length penalty")

    parser.add_argument("--num_return_sequences", default=1, type=int, help="number of returned sequence for each instance")
    parser.add_argument("--top_p", default=None, type=float, help="keep the top tokens with cumulative probability >= top_p")
    parser.add_argument("--top_k", default=None, type=int, help="keep only top k tokens with highest probability")
    parser.add_argument("--do_sample", action="store_true", help="Do sampling")

    return parser