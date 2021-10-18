import torch
import os
import pdb
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import random
import argparse
from multiprocessing import Pool
import numpy as np

from led_summ.utils.compute_scores import (
    compute_scores,
    get_entities,
)
from tqdm import tqdm
from timeit import default_timer as timer
from rouge_score import rouge_scorer
import re
import spacy


def get_src_tgt_with_mask(
    mask_indices,
    truncated_doc,
    tokenizer,
    max_length_input,
    max_length_output,
    non_mask_ratio=0.5,
    unchanged_mask_indices=None,
):
    tgt = []
    cur_idx = 0
    # print(mask_indices, int(len(mask_indices) * non_mask_ratio))
    if unchanged_mask_indices is None:
        non_mask_indices = random.sample(
            list(mask_indices), int(len(mask_indices) * non_mask_ratio)
        )
    else:
        non_mask_indices = unchanged_mask_indices

    for i_d in range(len(truncated_doc)):
        for i_s in range(len(truncated_doc[i_d])):
            if cur_idx in mask_indices:
                tgt.append(truncated_doc[i_d][i_s])
                if cur_idx not in non_mask_indices:
                    truncated_doc[i_d][i_s] = tokenizer.mask_token
            cur_idx += 1

    src = []
    truncated_doc = " %s " % (tokenizer.convert_tokens_to_ids("<doc-sep>")).join(
        [" ".join(d) for d in truncated_doc]
    )
    truncated_doc += " %s" % (tokenizer.convert_tokens_to_ids("<doc-sep>"))
    src = tokenizer.encode(truncated_doc, max_length=max_length_input, truncation=True)
    tgt = " ".join(tgt)
    tgt = tokenizer.encode(tgt, max_length=max_length_output, truncation=True)
    assert len(src) < max_length_input or len(src) == max_length_input
    assert len(tgt) < max_length_output or len(tgt) == max_length_output
    return src, tgt


def select_sent_with_entities(entities, all_docs, num_sent, metric):
    sent_plain = [s for doc_list in all_docs for s in doc_list]
    entity_info = {
        k: {"max_score": 0, "indice": -1} for k in entities.keys() if entities[k] > 1
    }
    for i, sent in enumerate(sent_plain):
        for e in entity_info.keys():
            if e.lower() in sent["text"].lower():
                if sent[metric] > entity_info[e]["max_score"]:
                    entity_info[e]["max_score"] = sent[metric]
                    entity_info[e]["indice"] = i
    sorted_entities = [
        k for k, v in sorted(entities.items(), key=lambda item: item[1]) if v > 1
    ][::-1]
    selected_sent = set()
    for k in sorted_entities:
        selected_sent.add(entity_info[k]["indice"])
        # stop when desired number of sentences are selected
        if len(selected_sent) >= num_sent:
            break
    # if entities are not enough
    if len(selected_sent) < num_sent:
        scores_flat = [s[metric] for s in sent_plain]
        greedy_indices = np.argsort(scores_flat)[::-1]
        for s in greedy_indices:
            if s not in selected_sent:
                selected_sent.add(s)
                if len(selected_sent) >= num_sent:
                    break
    return sorted(list(selected_sent))


def preprocessing_entities(entities):
    new_entities = {}
    for ent in entities.keys():
        new_ent = ent.lower().replace("the", "").strip()
        new_ent = new_ent.replace("'s", "").strip()
        new_entities[new_ent] = new_entities.get(new_ent, 0) + entities[ent]
    return new_entities


def truncate_doc(all_docs, max_length_input, mask_ratio, non_mask_ratio):
    # Truncate the documents to desired
    truncated_doc = []
    truncated_scores = []
    for doc in all_docs:
        cur_idx = 0
        all_sent_text = []
        all_scores = []
        for s in doc:
            cur_idx += len(s["text"].split())
            # if add the new sentence exceeds the expected length limit, don't add it
            if cur_idx >= (
                (max_length_input * (1 + mask_ratio * (1 - non_mask_ratio)))
                // len(all_docs)
            ):
                break
            all_sent_text.append(s["text"] + " .")
            all_scores.append(s)
        truncated_doc.append(all_sent_text)
        truncated_scores.append(all_scores)
    return truncated_doc, truncated_scores


def process_single_data_with_scores(
    all_docs,
    tokenizer,
    max_length_input,
    max_length_output,
    mask_ratio,
    metric="pegasus_score",
    strategy="greedy",
    non_mask_ratio=0.5,
):
    """
    all docs: dictionary for a single cluster, like
        {
         'entities_pyramid': {'ent1':1,'ent2':2,...}
         'entities': {'ent1':1,'ent2':3,...}
         'data':[[{'text': 'I like apple',
                    'pegasus_score': 0.3,
                    'pyramid_rouge': 0.5},
                  {'text': 'And I also like banana',
                    'pegasus_score': 0.5,
                    'pyramid_rouge': 0.4}],
                  ...],
                 [{'text': ...,
                    'pegasus_score': 0.4,
                    'pyramid_rouge': 0.6},
                 ...],
                ...]
        }

    """
    # if with entity

    entities = all_docs["entities_pyramid"]
    all_docs = all_docs["data"]
    # Truncate the documents to desired
    truncated_doc, scores = truncate_doc(
        all_docs, max_length_input, mask_ratio, non_mask_ratio
    )
    total_num_sentences = sum([len(d) for d in truncated_doc])
    mask_sent_num = int(total_num_sentences * mask_ratio)
    if strategy == "greedy":  # can only greedy on one metric
        scores_flat = [s for s_list in scores for s in s_list[metric]]
        # sorted_indices = np.argsort(scores_flat)[::-1]
        mask_indices = np.argsort(scores_flat)[::-1][:mask_sent_num]
    elif strategy == "greedy_entity_pyramid":
        mask_indices = select_sent_with_entities(
            entities, scores, mask_sent_num, metric
        )

    src, tgt = get_src_tgt_with_mask(
        mask_indices,
        truncated_doc,
        tokenizer,
        max_length_input,
        max_length_output,
        non_mask_ratio,
    )
    return {"src": src, "tgt": tgt}


def compute_all_scores_single_data(all_docs, i):
    cluster = [
        [
            s.strip()
            for p in re.split("\n+", doc)
            # for s in re.split(r"\.|\?|!", p)
            for s in sent_tokenize(p)
            if s.strip() != ""
        ]
        for doc in all_docs
    ]
    # take much time process very long sequence.
    if sum([len(d) for d in cluster]) > 600:
        return [
            [{"text": s, "pegasus_score": 0, "pyramid_rouge": 0} for s in d]
            for d in cluster
        ]
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    data_with_scores = compute_scores(cluster, scorer)

    # update entities
    result = update_entities(data_with_scores)

    return result


def update_entities(single_data_with_scores):
    nlp = spacy.load("en_core_web_sm")
    entity_pyramid, entity = get_entities(nlp, single_data_with_scores)
    result = {
        "data": single_data_with_scores,
        "entities": entity,
        "entities_pyramid": entity_pyramid,
    }
    return result


def process_all_newshead(
    input_dir,
    tokenizer,
    max_length_input,
    max_length_output,
    mask_ratio,
    output_dir,
    data_splits,
    num_workers,
    mode="compute_all_scores",
    metric="pegasus_score",
    strategy="greedy",
    non_mask_ratio=0.5,
):
    """
    Generate the pretraining data for wikisum dataset
    """
    if isinstance(data_splits, str):
        data_splits = [data_splits]
    for data_type in data_splits:
        all_files = [
            f
            for f in os.listdir(os.path.join(input_dir, data_type))
            if f.endswith(".pt")
        ]
        all_files = all_files[::-1]
        if mode == "compute_all_scores":
            out_path = os.path.join(output_dir, data_type)
        elif mode == "pretraining_data_with_score":
            out_path = os.path.join(
                output_dir,
                "%s_%s_%.1f_%.1f" % (strategy, metric, mask_ratio, non_mask_ratio),
                data_type,
            )
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print(out_path)
        for i_f, f in enumerate(tqdm(all_files)):
            if os.path.exists(os.path.join(out_path, f)):
                continue
            time_start = timer()
            all_data = torch.load(os.path.join(input_dir, data_type, f))
            if num_workers > 1:
                with Pool(num_workers) as processor:
                    if mode == "compute_all_scores":
                        inputs = [
                            ([doc["text"] for doc in d["articles"]], i)
                            for i, d in enumerate(all_data)
                        ]
                        new_data = processor.starmap(
                            compute_all_scores_single_data, inputs
                        )

                    elif mode == "pretraining_data_with_score":
                        inputs = [
                            (
                                d,
                                tokenizer,
                                max_length_input,
                                max_length_output,
                                mask_ratio,
                                metric,
                                strategy,
                                non_mask_ratio,
                            )
                            for i, d in enumerate(all_data)
                        ]
                        new_data = processor.starmap(
                            process_single_data_with_scores, inputs
                        )
            else:
                new_data = []
                for i, d in enumerate(all_data):
                    if mode == "compute_all_scores":
                        new_data.append(
                            compute_all_scores_single_data(
                                [doc["text"] for doc in d["articles"]], i
                            )
                        )

                    elif mode == "pretraining_data_with_score":
                        new_data.append(
                            process_single_data_with_scores(
                                d,
                                tokenizer,
                                max_length_input,
                                max_length_output,
                                mask_ratio,
                                metric,
                                strategy,
                                non_mask_ratio,
                            )
                        )

            torch.save(new_data, os.path.join(out_path, f))
            time_end = timer()
            print("%s saved! %d/%d finish!" % (f, i_f + 1, len(all_files)))
            print("finish one file, time is %f" % (time_end - time_start))
            # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Gneral
    parser.add_argument("--max_length_input", default=4096, type=int)
    parser.add_argument("--max_length_output", default=1024, type=int)
    parser.add_argument("--mask_ratio", default=0.3, type=float)
    parser.add_argument("--non_mask_ratio", default=0.5, type=float)
    parser.add_argument("--primer_path", default="./PRIMER", type=str)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data_with_scores/",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./dataset/newshead/",
    )
    parser.add_argument(
        "--data_splits",
        default="train",
        type=str,
        nargs="+",
        choices=["train", "valid", "test"],
    )
    parser.add_argument("--num_worker", default=1, type=int)
    parser.add_argument(
        "--mode",
        choices=[
            "compute_all_scores",
            "pretraining_data_with_score",
        ],
        default="pretraining_data",
        type=str,
    )

    parser.add_argument(
        "--metric",
        choices=["pegasus_score", "pyramid_rouge"],
        default="pegasus_score",
        type=str,
    )
    parser.add_argument(
        "--strategy",
        choices=[
            "greedy",
            "greedy_entity_pyramid",
        ],
        default="greedy",
        type=str,
    )

    args = parser.parse_args()
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.primer_path)

    output_dir = os.path.join(args.output_dir, "newshead")
    process_all_newshead(
        args.input_dir,
        tokenizer,
        args.max_length_input,
        args.max_length_output,
        args.mask_ratio,
        output_dir,
        args.data_splits,
        args.num_worker,
        args.mode,
        args.metric,
        args.strategy,
        args.non_mask_ratio,
    )
