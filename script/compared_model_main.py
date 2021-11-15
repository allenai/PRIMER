from numpy import NaN
import torch
import os
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor
from tqdm import tqdm

import pandas as pd
import pdb
from datasets import load_dataset, load_metric
import json
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    PegasusConfig,
    BigBirdPegasusForConditionalGeneration,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizerFast,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    LEDTokenizer,
    LEDForConditionalGeneration,
)
from transformers.file_utils import cached_path
from dataloader import (
    get_dataloader_summ,
    get_dataloader_summiter,
)
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pathlib import Path


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss


class BSLSummarizerLN(pl.LightningModule):
    def __init__(self, args):
        super(BSLSummarizerLN, self).__init__()
        self.args = args

        model_name = args.model_name
        if model_name == "google/pegasus-large":
            self.tokenizer = PegasusTokenizer.from_pretrained(
                model_name,
                cache_dir=os.path.join(args.pretrained_model_path, "pegasus"),
                model_max_len=args.max_length_input,
            )
            config = PegasusConfig.from_pretrained(
                model_name,
                cache_dir=os.path.join(args.pretrained_model_path, "pegasus"),
            )
            config.gradient_checkpointing = False
            config.max_position_embeddings = args.max_length_input
            self.model = PegasusForConditionalGeneration.from_pretrained(
                model_name,
                config=config,
                cache_dir=os.path.join(args.pretrained_model_path, "pegasus"),
            )
        elif model_name == "google/bigbird-pegasus-large-arxiv":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/bigbird-pegasus-large-arxiv",
                cache_dir=os.path.join(
                    args.pretrained_model_path, "bigbird-pegasus-large-arxiv"
                ),
            )
            self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(
                "google/bigbird-pegasus-large-arxiv",
                cache_dir=os.path.join(
                    args.pretrained_model_path, "bigbird-pegasus-large-arxiv"
                ),
                gradient_checkpointing=True,
            )
        elif model_name == "facebook/bart-large":
            self.tokenizer = BartTokenizerFast.from_pretrained(
                model_name,
                cache_dir=os.path.join(args.pretrained_model_path, "bart"),
                model_max_len=args.max_length_input,
            )
            config = BartConfig.from_pretrained(
                model_name,
                cache_dir=os.path.join(args.pretrained_model_path, "bart"),
            )
            config.gradient_checkpointing = True
            config.max_position_embeddings = args.max_length_input
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name,
                config=config,
                cache_dir=os.path.join(args.pretrained_model_path, "bart"),
            )
        elif model_name == "allenai/led-large-16384-arxiv":
            self.tokenizer = LEDTokenizer.from_pretrained(
                model_name,
                cache_dir=os.path.join(args.pretrained_model_path, "led_arxiv"),
            )
            self.model = LEDForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=os.path.join(args.pretrained_model_path, "led_arxiv"),
                gradient_checkpointing=True,
            )
        elif model_name == "allenai/led-large-16384":
            self.tokenizer = LEDTokenizer.from_pretrained(
                model_name,
                cache_dir=os.path.join(args.pretrained_model_path, "led_large"),
            )
            self.model = LEDForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=os.path.join(args.pretrained_model_path, "led_large"),
                gradient_checkpointing=True,
            )
        # if args.debug_mode:
        #     pdb.set_trace()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.use_ddp = args.accelerator == "ddp"

    def forward(self, input_ids, output_ids):
        # input_ids=batch.src
        # output_ids=batch.tgt

        # pdb.set_trace()
        decoder_input_ids = output_ids[:, :-1]
        if (
            self.args.model_name == "allenai/led-large-16384-arxiv"
            or self.args.model_name == "allenai/led-large-16384"
        ):
            # get the input ids and attention masks together
            global_attention_mask = torch.zeros_like(input_ids).cuda()
            # put global attention on <s> token

            global_attention_mask[:, 0] = 1
            outputs = self.model(
                input_ids,
                decoder_input_ids=decoder_input_ids,
                global_attention_mask=global_attention_mask,
                use_cache=False,
            )
        else:

            # decoder_attention_mask = decoder_input_ids != self.pad_token_id
            # pdb.set_trace()
            outputs = self.model(
                input_ids,
                decoder_input_ids=decoder_input_ids,
                # decoder_attention_mask=decoder_attention_mask,
                use_cache=False,
            )
        lm_logits = outputs[0]
        # pdb.set_trace()
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        return lm_logits

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(
                self.parameters(),
                lr=self.args.lr,
                scale_parameter=False,
                relative_step=False,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.args.warmup_steps
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.total_steps,
            )
        if self.args.fix_lr:
            return optimizer
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def shared_step(self, input_ids, output_ids):
        lm_logits = self.forward(input_ids, output_ids)
        labels = output_ids[:, 1:].clone()
        # pdb.set_trace()

        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                labels,
                self.args.label_smoothing,
                ignore_index=self.pad_token_id,
            )
        if torch.isnan(loss):
            pdb.set_trace()
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # pdb.set_trace()
        # if self.args.debug_mode:
        #     return None
        input_ids, output_ids = batch
        loss = self.shared_step(input_ids, output_ids)
        # if torch.isnan(loss):
        #     pdb.set_trace()
        #     loss = self.shared_step(input_ids, output_ids)

        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]["lr"]
        tensorboard_logs = {
            "train_loss": loss,
            "lr": lr,
            "input_size": input_ids.numel(),
            "output_size": output_ids.numel(),
            "mem": torch.cuda.memory_allocated(loss.device) / 1024 ** 3
            if torch.cuda.is_available()
            else 0,
        }
        self.logger.log_metrics(tensorboard_logs, step=self.global_step)
        # pdb.set_trace()
        return loss

    def compute_rouge_batch(self, input_ids, output_ids, gold_str):
        scorer = load_metric("rouge")
        # pdb.set_trace()
        if (
            self.args.model_name == "allenai/led-large-16384-arxiv"
            or self.args.model_name == "allenai/led-large-16384"
        ):
            global_attention_mask = torch.zeros_like(input_ids).cuda()
            # put global attention on <s> token
            global_attention_mask[:, 0] = 1
            generated_ids = self.model.generate(
                input_ids=input_ids,
                # attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                use_cache=True,
                max_length=self.args.max_length_tgt,
                min_length=self.args.min_length_tgt,
                num_beams=self.args.beam_size,
                no_repeat_ngram_size=3 if self.args.applyTriblck else None,
                length_penalty=self.args.length_penalty,
            )
        else:
            generated_ids = self.model.generate(
                input_ids=input_ids,
                use_cache=True,
                max_length=self.args.max_length_tgt,
                min_length=self.args.min_length_tgt,
                num_beams=self.args.beam_size,
                no_repeat_ngram_size=3 if self.args.applyTriblck else None,
                length_penalty=self.args.length_penalty,
            )
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )

        # gold_str = self.tokenizer.batch_decode(
        #     output_ids.tolist(), skip_special_tokens=True
        # )

        if self.args.mode == "test":
            if self.args.applyTriblck:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generated_txt_%d_%s_triblck_beam=%d_%d_%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.beam_size,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                    ),
                )
            else:
                output_dir = os.path.join(
                    self.args.model_path,
                    "generated_txt_%d_%s_beam=%d_%d_%d"
                    % (
                        self.args.mask_num,
                        self.args.dataset_name,
                        self.args.beam_size,
                        self.args.max_length_input,
                        self.args.max_length_tgt,
                    ),
                )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            idx = len(os.listdir(output_dir))
        result_batch = []
        if self.args.debug_mode:
            pdb.set_trace()
        for ref, pred in zip(gold_str, generated_str):
            # change <n> to \n
            pred = pred.replace("<n>", "\n")

            if self.args.mode == "test":
                with open(os.path.join(output_dir, "%d.txt" % (idx)), "w") as of:
                    of.write(pred)
                idx += 1

            s = scorer.compute(
                predictions=[pred],
                references=[ref],
                use_agregator=False,
                use_stemmer=True,
            )
            result_batch.append(
                (
                    s["rouge1"][0].recall,
                    s["rouge1"][0].precision,
                    s["rouge1"][0].fmeasure,
                    s["rouge2"][0].recall,
                    s["rouge2"][0].precision,
                    s["rouge2"][0].fmeasure,
                    s["rougeL"][0].recall,
                    s["rougeL"][0].precision,
                    s["rougeL"][0].fmeasure,
                    s["rougeLsum"][0].recall,
                    s["rougeLsum"][0].precision,
                    s["rougeLsum"][0].fmeasure,
                )
            )

            # pdb.set_trace()
        return result_batch

    def validation_step(self, batch, batch_idx):
        for p in self.model.parameters():
            p.requires_grad = False
        input_ids, output_ids, tgt = batch
        loss = self.shared_step(input_ids, output_ids)
        if self.args.compute_rouge:
            result_batch = self.compute_rouge_batch(input_ids, output_ids, tgt)
            return {"vloss": loss, "rouge_result": result_batch}
        else:
            return {"vloss": loss}

    def compute_rouge_all(self, outputs, output_file=None):
        rouge_result_all = [r for b in outputs for r in b["rouge_result"]]
        names = []
        for rouge in ["1", "2", "L", "Lsum"]:
            names.extend(
                [
                    "rouge-{}-r".format(rouge),
                    "rouge-{}-p".format(rouge),
                    "rouge-{}-f".format(rouge),
                ]
            )
        rouge_results = pd.DataFrame(rouge_result_all, columns=names)
        avg = [rouge_results[c].mean() for c in rouge_results.columns]
        rouge_results.loc["avg_score"] = avg
        if output_file:
            csv_name = (
                args.model_path
                + output_file
                + "-%d.csv" % (torch.distributed.get_rank() if self.use_ddp else 0)
            )
            rouge_results.to_csv(csv_name)

        avgr = (avg[2] + avg[5] + avg[8]) / 3
        metrics = avg
        print("Validation Result at Step %d" % (self.global_step))
        print(
            "Rouge-1 r score: %f, Rouge-1 p score: %f, Rouge-1 f-score: %f"
            % (metrics[0], metrics[1], metrics[2])
        )
        print(
            "Rouge-2 r score: %f, Rouge-2 p score: %f, Rouge-2 f-score: %f"
            % (metrics[3], metrics[4], metrics[5])
        )
        print(
            "Rouge-L r score: %f, Rouge-L p score: %f, Rouge-L f-score: %f"
            % (metrics[6], metrics[7], metrics[8])
        )
        print(
            "Rouge-Lsum r score: %f, Rouge-Lsum p score: %f, \
            Rouge-Lsum f-score: %f"
            % (metrics[9], metrics[10], metrics[11])
        )
        return names, metrics, avgr

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        vloss = torch.stack([x["vloss"] for x in outputs]).mean()
        self.log("vloss", vloss, sync_dist=True if self.use_ddp else False)
        if self.args.compute_rouge:
            names, metrics, avgr = self.compute_rouge_all(outputs, output_file="valid")
            metrics = [vloss] + metrics
            names = ["vloss"] + names
            logs = dict(zip(*[names, metrics]))
            self.logger.log_metrics(logs, step=self.global_step)
            self.log("avgr", avgr)
            return {
                "avg_val_loss": vloss,
                "avgr": avgr,
                "log": logs,
                "progress_bar": logs,
            }
        else:
            logs = {"vloss": vloss}
            self.logger.log_metrics(logs, step=self.global_step)
            return {"vloss": vloss, "log": logs, "progress_bar": logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        tloss = torch.stack([x["vloss"] for x in outputs]).mean()
        self.log("tloss", tloss, sync_dist=True if self.use_ddp else False)
        output_file = "test_%s_%d_%d_beam=%d_lenPen=%.2f" % (
            self.args.dataset_name,
            self.args.max_length_input,
            self.args.max_length_tgt,
            self.args.beam_size,
            self.args.length_penalty,
        )
        output_file = (
            output_file
            + "_fewshot_%d_%d" % (self.args.num_train_data, self.args.rand_seed)
            if self.args.fewshot
            else output_file
        )
        names, metrics, avgr = self.compute_rouge_all(outputs, output_file=output_file)
        metrics = [tloss, avgr] + metrics
        names = ["tloss", "avgr"] + names
        logs = dict(zip(*[names, metrics]))
        self.logger.log_metrics(logs, step=self.global_step)
        self.log("avgr", avgr)
        # self.log_dict(logs)
        return {"avg_test_loss": tloss, "avgr": avgr, "log": logs, "progress_bar": logs}


def train(args):
    args.compute_rouge = True
    model = BSLSummarizerLN(args)

    # initialize checkpoint
    if args.ckpt_path is None:
        args.ckpt_path = args.model_path + "summ_checkpoints/"

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="{step}-{vloss:.2f}-{avgr:.4f}",
        save_top_k=args.saveTopK,
        monitor="avgr",
        mode="max",
        save_on_train_epoch_end=False,
    )

    # initialize logger
    logger = TensorBoardLogger(args.model_path + "tb_logs", name="my_model")

    # initialize trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        track_grad_norm=-1,
        max_steps=args.total_steps,
        replace_sampler_ddp=False,
        accumulate_grad_batches=args.acc_batch,
        # val_check_interval=0.5,
        check_val_every_n_epoch=1 if args.num_train_data > 100 else 5,
        logger=logger,
        log_every_n_steps=5,
        callbacks=checkpoint_callback,
        checkpoint_callback=True,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate * args.acc_batch,
        precision=32,
        accelerator=args.accelerator,
    )

    # load datasets
    if args.dataset_name in ["multi_news", "multi_x_science_sum"]:

        hf_datasets = load_dataset(args.dataset_name, cache_dir=args.data_path)
        train_dataloader = get_dataloader_summ(
            args, hf_datasets, model.tokenizer, "train", 0, True
        )
        valid_dataloader = get_dataloader_summ(
            args, hf_datasets, model.tokenizer, "validation", 0, False
        )
    elif (
        ("duc" in args.dataset_name)
        or ("tac" in args.dataset_name)
        or args.dataset_name == "wcep"
        or args.dataset_name == "wikisum"
    ):
        # 20 data from duc2003
        dataset = torch.load(args.data_path + "train.pt")
        train_dataloader = get_dataloader_summ(
            args, dataset, model.tokenizer, "train", 0, True
        )
        # 10 data from duc2003
        if os.path.exists(args.data_path + "val.pt"):
            dataset = torch.load(args.data_path + "val.pt")
        else:
            dataset = torch.load(args.data_path + "valid.pt")
        valid_dataloader = get_dataloader_summ(
            args, dataset, model.tokenizer, "validation", 0, True
        )
    elif args.dataset_name == "arxiv":
        with open(args.data_path + "train.txt", "r") as of:
            all_lines = of.readlines()
        dataset = [json.loads(l) for l in all_lines]
        train_dataloader = get_dataloader_summ(
            args, dataset, model.tokenizer, "train", 0, False
        )
        with open(args.data_path + "val.txt", "r") as of:
            all_lines = of.readlines()
        dataset = [json.loads(l) for l in all_lines]
        valid_dataloader = get_dataloader_summ(
            args, dataset, model.tokenizer, "validation", 0, False
        )
    # pdb.set_trace()
    trainer.fit(model, train_dataloader, valid_dataloader)
    if args.test_imediate:
        args.resume_ckpt = checkpoint_callback.best_model_path
        print(args.resume_ckpt)
        if args.test_batch_size != -1:
            args.batch_size = args.test_batch_size
        args.mode = "test"
        test(args)


def test(args):
    args.compute_rouge = True
    # initialize trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        track_grad_norm=-1,
        max_steps=args.total_steps * args.acc_batch,
        replace_sampler_ddp=False,
        log_every_n_steps=5,
        checkpoint_callback=True,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        precision=32,
        accelerator=args.accelerator,
        limit_test_batches=args.limit_test_batches if args.limit_test_batches else 1.0,
    )

    if args.resume_ckpt is not None:
        model = BSLSummarizerLN.load_from_checkpoint(args.resume_ckpt, args=args)
    else:
        model = BSLSummarizerLN(args)

    # load dataset
    if args.dataset_name in ["multi_news", "multi_x_science_sum"]:

        hf_datasets = load_dataset(args.dataset_name, cache_dir=args.data_path)
        test_dataloader = get_dataloader_summ(
            args, hf_datasets, model.tokenizer, "test", 0, False
        )
    elif (
        ("duc" in args.dataset_name)
        or ("tac" in args.dataset_name)
        or args.dataset_name == "wcep"
        or args.dataset_name == "wikisum"
    ):
        if os.path.isdir(args.data_path):
            dataset = torch.load(args.data_path + "test.pt")
        else:
            dataset = torch.load(args.data_path)
        test_dataloader = get_dataloader_summ(
            args, dataset, model.tokenizer, "test", 0, False
        )
    elif args.dataset_name == "arxiv":
        with open(args.data_path + "test.txt", "r") as of:
            all_lines = of.readlines()
        dataset = [json.loads(l) for l in all_lines]
        test_dataloader = get_dataloader_summ(
            args, dataset, model.tokenizer, "test", 0, False
        )

    # test
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ########################
    # Gneral
    parser.add_argument("--gpus", default=0, type=int, help="number of gpus to use")
    parser.add_argument(
        "--accelerator", default=None, type=str, help="Type of accelerator"
    )
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument(
        "--model_name",
        default="google/pegasus-large",
        choices=[
            "google/pegasus-large",
            "google/bigbird-pegasus-large-arxiv",
            "facebook/bart-large",
            "allenai/led-large-16384-arxiv",
            "allenai/led-large-16384",
        ],
    )
    parser.add_argument("--join_method", type=str, default="concat_start_eachdoc")
    parser.add_argument(
        "--debug_mode", action="store_true", help="set true if to debug"
    )
    parser.add_argument(
        "--compute_rouge",
        action="store_true",
        help="whether to compute rouge in validation steps",
    )
    parser.add_argument(
        "--saveRouge",
        action="store_true",
        help="whether to compute rouge in validation steps",
    )

    parser.add_argument("--progress_bar_refresh_rate", default=1, type=int)
    parser.add_argument("--model_path", type=str, default="./pegasus/")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--saveTopK", default=3, type=int)
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        help="Path of a checkpoint to resume from",
        default=None,
    )

    parser.add_argument("--data_path", type=str, default="../dataset/")
    parser.add_argument("--dataset_name", type=str, default="arxiv")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use for dataloader",
    )

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_length_input", default=4096, type=int)
    parser.add_argument("--max_length_tgt", default=1024, type=int)
    parser.add_argument("--min_length_tgt", default=0, type=int)
    parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
    parser.add_argument(
        "--adafactor", action="store_true", help="Use adafactor optimizer"
    )
    parser.add_argument(
        "--grad_ckpt",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=0,
        help="seed for random sampling, useful for few shot learning",
    )

    ########################
    # For training
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="./pretrained_models/",
    )
    parser.add_argument(
        "--limit_valid_batches",
        type=int,
        default=None,
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Maximum learning rate")
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--accum_data_per_step", type=int, default=16, help="Number of data per step"
    )
    parser.add_argument(
        "--total_steps", type=int, default=50000, help="Number of steps to train"
    )
    parser.add_argument(
        "--num_train_data",
        type=int,
        default=-1,
        help="Number of training data, -1 for full dataset and any positive number indicates how many data to use",
    )

    parser.add_argument(
        "--fix_lr",
        action="store_true",
        help="use fix learning rate",
    )
    parser.add_argument(
        "--test_imediate",
        action="store_true",
        help="test on the best checkpoint",
    )
    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="whether this is a run for few shot learning",
    )
    ########################
    # For testing
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="Number of batches to test in the test mode.",
    )
    parser.add_argument("--beam_size", type=int, default=1, help="size of beam search")
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1,
        help="length penalty of generated text",
    )
    parser.add_argument(
        "--mask_num",
        type=int,
        default=0,
        help="Number of masks in the input of summarization data",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=-1,
        help="batch size for test, used in few shot evaluation.",
    )
    parser.add_argument(
        "--applyTriblck",
        action="store_true",
        help="whether apply trigram block in the evaluation phase",
    )

    args = parser.parse_args()  # Get pad token id
    ####################
    args.acc_batch = args.accum_data_per_step // args.batch_size
    args.data_path = os.path.join(args.data_path, args.dataset_name)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print(args)
    with open(
        os.path.join(
            args.model_path, "args_%s_%s.json" % (args.mode, args.dataset_name)
        ),
        "w",
    ) as f:
        json.dump(args.__dict__, f, indent=2)

    if args.mode == "train":
        train(args)
    else:

        test(args)
