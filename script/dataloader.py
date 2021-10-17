from torch.utils.data import DataLoader, Dataset, IterableDataset
from pathlib import Path
import torch
from random import shuffle
import random
import os
from nltk.tokenize import sent_tokenize
import re
import sys


class SummarizationDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        dataset_name,
        join_method,
        tokenizer,
        max_input_len,
        max_output_len,
        mask_num=5,
        num_data=-1,
        rand_seed=1,
        is_test=False,
        dataset_type="train",
    ):
        self.hf_dataset = hf_dataset
        self.dataset_name = dataset_name
        self.join_method = join_method
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        if join_method == "concat_start_wdoc_global":
            self.docsep_token_id = self.tokenizer.additional_special_tokens_ids[0]
        self.mask_id = self.tokenizer.mask_token_id
        self.mask_num = mask_num
        if num_data != -1 and not is_test and num_data < len(list(hf_dataset)):
            random.seed(rand_seed)
            self.hf_dataset = random.sample(list(hf_dataset), num_data)
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        # single doc setting
        if self.dataset_name == "pubmed":
            src = entry["article"]
            tgt = entry["abstract"]
            input_ids = self.tokenizer.encode(
                src, truncation=True, max_length=self.max_input_len
            )
            output_ids = self.tokenizer.encode(
                tgt, truncation=True, max_length=self.max_output_len
            )
        else:  # multi-doc setting
            if self.dataset_name == "multi_news":
                all_docs = entry["document"].split("|||||")[:-1]
                for i, doc in enumerate(all_docs):
                    doc = doc.replace("\n", " ")
                    doc = " ".join(doc.split())
                    all_docs[i] = doc
                tgt = entry["summary"]
            elif self.dataset_name == "multi_x_science_sum":
                all_docs = [entry["abstract"]]
                for d in entry["ref_abstract"]["abstract"]:
                    if len(d) > 0:
                        all_docs.append(d)
                tgt = entry["related_work"]
                # remove all @cite_d
                tgt = re.sub(r"\@cite_\d+", "cite", tgt)

            elif ("duc" in self.dataset_name) or ("tac" in self.dataset_name):
                all_docs = entry["document"]
                tgt = entry["summary"][0]  # simply use the first gt summary
            elif self.dataset_name == "wcep" or self.dataset_name == "arxiv":
                all_docs = entry["document"]
                tgt = entry["summary"]
            elif self.dataset_name == "wikisum":
                all_docs = entry["text"]
                tgt = entry["tgt"]
            if self.join_method == "plain_concat":
                src = "\n".join(all_docs)
                input_ids = self.tokenizer.encode(
                    src, truncation=True, max_length=self.max_input_len
                )
            elif self.join_method == "concat_start_eachdoc":
                input_text = []
                for doc in all_docs:
                    length = 0
                    all_sents = sent_tokenize(doc)
                    for s in all_sents:
                        input_text.append(s)
                        length += len(s.split())
                        if length >= self.max_input_len // len(all_docs):
                            break
                input_ids = self.tokenizer.encode(
                    " ".join(input_text),
                    truncation=True,
                    max_length=self.max_input_len,
                )
            elif self.join_method == "concat_start_eachdoc_wsent_global":
                input_ids = []
                for doc in all_docs:
                    sents = [
                        " [sent] ".join(sent_tokenize(p)) + " [sent]"
                        for p in doc.split("\n")
                        if p != ""
                    ]
                    doc = "\n".join(sents)
                    input_ids.extend(
                        self.tokenizer.encode(
                            doc,
                            truncation=True,
                            max_length=self.max_input_len // len(all_docs),
                        )[1:-1]
                    )
                input_ids = (
                    [self.tokenizer.bos_token_id]
                    + input_ids
                    + [self.tokenizer.eos_token_id]
                )
            elif self.join_method == "concat_start_wdoc_global":
                mask_num = self.mask_num

                input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                for doc in all_docs:
                    input_ids.extend(
                        self.tokenizer.encode(
                            doc,
                            truncation=True,
                            max_length=(self.max_input_len - mask_num) // len(all_docs),
                        )[1:-1]
                    )
                    input_ids.append(self.docsep_token_id)
                input_ids = (
                    [self.tokenizer.bos_token_id]
                    + input_ids
                    + [self.tokenizer.eos_token_id]
                )

            output_ids = self.tokenizer.encode(
                tgt, truncation=True, max_length=self.max_output_len
            )

        if self.tokenizer.bos_token_id is None:  # pegasus
            output_ids = [self.tokenizer.pad_token_id] + output_ids
        if self.dataset_type == "train":
            return torch.tensor(input_ids), torch.tensor(output_ids)
        else:
            return torch.tensor(input_ids), torch.tensor(output_ids), tgt


class PretrainDataset(IterableDataset):
    def __init__(
        self,
        inputs_dir,
        dataset_type,
        max_input_len,
        max_output_len,
        use_ddp=False,
        remove_masks=False,
        mask_id=0,
    ):
        super().__init__()
        if isinstance(inputs_dir, list):
            self._input_files = inputs_dir
        else:
            inputs_dir = Path(os.path.join(inputs_dir, dataset_type))
            self._input_files = [path for path in inputs_dir.glob("*.pt")]
        self.shuffle = dataset_type == "train"
        self._input_files = sorted(self._input_files)
        if self.shuffle:
            self._shuffle()
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.start = 0
        self.end = len(self._input_files)
        self.use_ddp = use_ddp
        self.remove_masks = remove_masks
        self.mask_id = mask_id

    def _loaddata(self, idx):
        file = self._input_files[idx]
        cur_data = torch.load(file)
        if self.shuffle:
            shuffle(cur_data)
        return cur_data

    def _shuffle(self):
        # shuffle the list of data files after each epoch
        shuffle(self._input_files)

    def _set_worker(self):
        # The whole dataset covering all the files in self._input_files
        overall_start = 0
        overall_end = len(self._input_files)

        # Get the worker id in the current world
        worker_info = torch.utils.data.get_worker_info()
        num_workers, worker_id = (
            (worker_info.num_workers, worker_info.id)
            if worker_info is not None
            else (1, 0)
        )

        # Get the worker id in the overall worlds
        if self.use_ddp:
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            worker_global_rank = global_rank * num_workers + worker_id
        else:
            worker_global_rank = worker_id
            world_size = 1

        # Get the total number of workers and split tasks accordingly
        worker_world_size = num_workers * world_size
        per_worker = int(
            math.ceil((overall_end - overall_start) / float(worker_world_size))
        )

        # Set the current task, based on overall worker id and task splitting
        self.start = overall_start + worker_global_rank * per_worker
        self.end = min(self.start + per_worker, overall_end)

    def __iter__(self):
        self._set_worker()
        all_indices = list(range(self.start, self.end))
        if self.shuffle:
            # use inifinite iterators for training
            while True:
                shuffle(all_indices)
                for i in all_indices:
                    print("datafile is ", self._input_files[i])
                    sys.stdout.flush()
                    cur_data = self._loaddata(i)
                    while len(cur_data) != 0:
                        # print('data index is ', len(cur_data))
                        data = cur_data.pop()
                        if self.remove_masks:
                            data["src"] = list(
                                filter(lambda a: a != self.mask_id, data["src"])
                            )
                            # print(data["src"])
                        if len(data["src"]) > self.max_input_len:
                            data["src"] = data["src"][: (self.max_input_len - 1)] + [
                                data["src"][-1]
                            ]  # add </s>
                        if len(data["tgt"]) > self.max_output_len:
                            data["tgt"] = data["tgt"][: (self.max_output_len - 1)] + [
                                data["tgt"][-1]
                            ]  # add </s>
                        yield torch.tensor(data["src"]), torch.tensor(data["tgt"])
        else:
            # use normal iterators for validation
            for i in all_indices:
                print("datafile is ", self._input_files[i])
                sys.stdout.flush()
                cur_data = self._loaddata(i)
                while len(cur_data) != 0:
                    # print('data index is ', len(cur_data))
                    data = cur_data.pop()
                    if self.remove_masks:
                        data["src"] = list(
                            filter(lambda a: a != self.mask_id, data["src"])
                        )
                        # print(data["src"])
                    if len(data["src"]) > self.max_input_len:
                        data["src"] = data["src"][: (self.max_input_len - 1)] + [
                            data["src"][-1]
                        ]  # add </s>
                    if len(data["tgt"]) > self.max_output_len:
                        data["tgt"] = data["tgt"][: (self.max_output_len - 1)] + [
                            data["tgt"][-1]
                        ]  # add </s>
                    yield torch.tensor(data["src"]), torch.tensor(data["tgt"])


class SummarizationIterDataset(IterableDataset):
    def __init__(
        self,
        join_method,
        dataset_name,
        tokenizer,
        inputs_dir,
        dataset_type,
        max_input_len,
        max_output_len,
        use_ddp=False,
        mask_num=0,
    ):
        super().__init__()
        if isinstance(inputs_dir, list):
            self._input_files = inputs_dir
        else:
            inputs_dir = Path(os.path.join(inputs_dir, dataset_type))
            self._input_files = [path for path in inputs_dir.glob("*.pt")]
        self._input_files = sorted(self._input_files)
        self.shuffle = dataset_type == "train"
        if self.shuffle:
            self._shuffle()
        self.join_method = join_method
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.start = 0
        self.end = len(self._input_files)
        self.use_ddp = use_ddp
        if join_method == "concat_start_wdoc_global":
            self.docsep_token_id = self.tokenizer.additional_special_tokens_ids[0]
        self.mask_num = mask_num
        self.dataset_type = dataset_type

    def _loaddata(self, idx):
        file = self._input_files[idx]
        cur_data = torch.load(file)
        if self.shuffle:
            shuffle(cur_data)
        return cur_data

    def _shuffle(self):
        # shuffle the list of data files after each epoch
        shuffle(self._input_files)

    def _set_worker(self):
        # The whole dataset covering all the files in self._input_files
        overall_start = 0
        overall_end = len(self._input_files)

        # Get the worker id in the current world
        worker_info = torch.utils.data.get_worker_info()
        num_workers, worker_id = (
            (worker_info.num_workers, worker_info.id)
            if worker_info is not None
            else (1, 0)
        )

        # Get the worker id in the overall worlds
        if self.use_ddp:
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            worker_global_rank = global_rank * num_workers + worker_id
        else:
            worker_global_rank = worker_id
            world_size = 1

        # Get the total number of workers and split tasks accordingly
        worker_world_size = num_workers * world_size
        per_worker = int(
            math.ceil((overall_end - overall_start) / float(worker_world_size))
        )

        # Set the current task, based on overall worker id and task splitting
        self.start = overall_start + worker_global_rank * per_worker
        self.end = min(self.start + per_worker, overall_end)

    def __iter__(self):
        self._set_worker()

        for i in range(self.start, self.end):
            print("datafile is ", i)
            cur_data = self._loaddata(i)
            while len(cur_data) != 0:
                # print("data index is ", len(cur_data))
                data = cur_data.pop()
                all_docs = data["text"]
                if self.join_method == "plain_concat":
                    src = "\n".join(all_docs)
                    tgt = data["tgt"]
                    input_ids = self.tokenizer.encode(
                        src, truncation=True, max_length=self.max_input_len
                    )
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )
                elif self.join_method == "concat_start_eachdoc":
                    input_ids = []
                    for doc in all_docs:
                        input_ids.extend(
                            self.tokenizer.encode(
                                doc,
                                truncation=True,
                                max_length=self.max_input_len // len(all_docs),
                            )[1:-1]
                        )
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                    tgt = data["tgt"]
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )
                elif self.join_method == "concat_start_eachdoc_wsent_global":
                    input_ids = []
                    for doc in all_docs:
                        sents = [
                            " [sent] ".join(sent_tokenize(p)) + " [sent]"
                            for p in doc.split("\n")
                            if p != ""
                        ]
                        doc = "\n".join(sents)
                        input_ids.extend(
                            self.tokenizer.encode(
                                doc,
                                truncation=True,
                                max_length=self.max_input_len // len(all_docs),
                            )[1:-1]
                        )
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                    tgt = data["tgt"]
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )
                elif self.join_method == "concat_start_wdoc_global":
                    mask_num = self.mask_num
                    tgt = data["tgt"]
                    # src='<mask>'*10+src
                    input_ids = [self.mask_id] * mask_num if mask_num > 0 else []
                    for doc in all_docs:
                        input_ids.extend(
                            self.tokenizer.encode(
                                doc,
                                truncation=True,
                                max_length=(self.max_input_len - mask_num)
                                // len(all_docs),
                            )[1:-1]
                        )
                        input_ids.append(self.docsep_token_id)
                    input_ids = (
                        [self.tokenizer.bos_token_id]
                        + input_ids
                        + [self.tokenizer.eos_token_id]
                    )
                    output_ids = self.tokenizer.encode(
                        tgt, truncation=True, max_length=self.max_output_len
                    )

                if self.tokenizer.bos_token_id is None:  # pegasus
                    output_ids = [self.tokenizer.pad_token_id] + output_ids
                    input_ids = input_ids[1:]
                if self.dataset_type == "train":
                    yield torch.tensor(input_ids), torch.tensor(output_ids)
                else:
                    yield torch.tensor(input_ids), torch.tensor(output_ids), tgt


def collate_fn(batch):
    # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
    if batch[0][0][-1].item() == 2:
        pad_token_id = (
            1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        )
    elif batch[0][0][-1].item() == 1:
        pad_token_id = (
            0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        )
    else:
        assert False
    train = True
    if len(batch[0]) == 3:
        train = False
        tgt = [item[2] for item in batch]
        batch = [item[:2] for item in batch]
    input_ids, output_ids = list(zip(*batch))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    output_ids = torch.nn.utils.rnn.pad_sequence(
        output_ids, batch_first=True, padding_value=pad_token_id
    )
    if train:
        return input_ids, output_ids
    else:
        return input_ids, output_ids, tgt


def get_dataloader_summ(
    args, hf_datasets, tokenizer, split_name, num_workers, is_train
):
    if (
        ("duc" in args.dataset_name)
        or ("tac" in args.dataset_name)
        or args.dataset_name == "wcep"
        or args.dataset_name == "wikisum"
    ):
        d = hf_datasets
    elif args.dataset_name == "arxiv":

        d = [
            {
                "document": [" ".join(s) for s in single_data["sections"]],
                "summary": " ".join(
                    [
                        sent.replace("<S>", "").replace("</S>", "").strip()
                        for sent in single_data["abstract_text"]
                    ]
                ),
            }
            for single_data in hf_datasets
        ]
    else:
        d = hf_datasets[split_name]
    dataset = SummarizationDataset(
        hf_dataset=d,
        dataset_name=args.dataset_name,
        join_method=args.join_method,
        tokenizer=tokenizer,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        mask_num=args.mask_num,
        num_data=args.num_train_data,
        rand_seed=args.rand_seed,
        is_test=(split_name == "test"),
        dataset_type=split_name,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        # sampler=sampler,
        collate_fn=collate_fn,
    )


def get_dataloader_pretrain(
    args, inputs_dir, dataset_type, num_workers, use_ddp=False, mask_id=0
):
    dataset = PretrainDataset(
        inputs_dir,
        dataset_type,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        use_ddp=use_ddp,
        remove_masks=args.remove_masks,
        mask_id=mask_id,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def get_dataloader_summiter(
    args, tokenizer, inputs_dir, dataset_type, num_workers, use_ddp=False
):
    dataset = SummarizationIterDataset(
        args.join_method,
        args.dataset_name,
        tokenizer,
        inputs_dir,
        dataset_type,
        max_input_len=args.max_length_input,
        max_output_len=args.max_length_tgt,
        use_ddp=use_ddp,
        mask_num=args.mask_num,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
