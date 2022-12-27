"""Filtering routines for APE."""
from copy import deepcopy
import multiprocessing as multi
import os
import random
import subprocess

from sacremoses import MosesTokenizer
from sentencepiece import SentencePieceProcessor

from corpus import Corpus


def read_through(corpus: Corpus, read_que: multi.Queue):
    """Read data from a corpus."""
    for line in corpus:
        read_que.put(line)
    read_que.put(None)


def tokenize(
    proced_info: dict, corpus_info: dict, line: dict,
):
    """Tokenize given texts."""
    for data_name in corpus_info:
        tok_script = proced_info["tokenization"]["German script"]
        if data_name == "source":
            tok_script = proced_info["tokenization"]["English script"]
        line[data_name] = tok_script.tokenize(
            line[data_name], return_str=True, escape=False)


def divide_into_subword_units(
    proced_info: dict, corpus_info: dict, line: dict,
):
    """Divide tokens into subword units."""
    for data_name in corpus_info:
        seg_sys = proced_info["subword segmentation"]["system"]
        line[data_name] = seg_sys.encode(
            line[data_name], out_type=str)


def concat_input(proced_info: dict, line: dict):
    """Concatenate a source text and its machine translation."""
    line["source"] = [
        *line["source"],
        "<s>",
        *line["mach_trans"],
    ]


def draw_out_samples(lines: list[dict], sample_size: int):
    """Draw out only a few samples."""
    shuffled_ind = list(range(len(lines)))
    random.shuffle(shuffled_ind)
    sampled_ind = shuffled_ind[:sample_size]
    samples = []
    for ind in sampled_ind:
        samples.append(lines[ind])
    return samples


def process_corpus(
    read_que: multi.Queue, writ_que: multi.Queue, corpus_info: dict,
    proced_info: dict,
):
    """Process an APE corpus."""
    while True:
        line = read_que.get()
        if line is None:
            break
        for data_name in corpus_info:
            if len(line[data_name]) < 1:
                break
        if "tokenization" in proced_info:
            tokenize(proced_info, corpus_info, line)
        if "subword segmentation" in proced_info:
            divide_into_subword_units(proced_info, corpus_info, line)
        if "length control" in proced_info:
            max_len = proced_info["length control"]["max"]
            too_long = False
            for data_name in corpus_info:
                if len(line[data_name]) > max_len:
                    too_long = True
                    break
            if too_long:
                continue
        if "input concatenation" in proced_info:
            concat_input(proced_info, line)
        writ_que.put(line)
    writ_que.put(None)


def record(writ_que: multi.Queue, out_files: dict, proced_info: dict):
    """Write out data."""
    if "draw out samples" in proced_info:
        lines = []
        while True:
            line = writ_que.get()
            if line is None:
                break
            lines.append(line)
        sample_size = proced_info["draw out samples"]["size"]
        samples = draw_out_samples(lines, sample_size)
        for line in samples:
            for data_name, out_file in out_files.items():
                if not isinstance(line[data_name], str):
                    writ_text = " ".join(line[data_name]) + "\n"
                else:
                    writ_text = line[data_name] + "\n"
                out_file.write(writ_text)
                out_file.flush()
    else:
        while True:
            line = writ_que.get()
            if line is None:
                break
            for data_name, out_file in out_files.items():
                writ_text = " ".join(line[data_name]) + "\n"
                out_file.write(writ_text)
                out_file.flush()


class APEFilter:
    """A basic filter routine for APE."""

    def __init__(self, corpus: Corpus, supp_info: dict):

        self.corpus = corpus
        self.supp_info = supp_info
        self.info = supp_info["pretreat"]["ape_filter"]

        self.corpus_info = self.info["corpora"][corpus.name]
        self.buff_dir = self.corpus_info.get(
            "buff_dir",
            {
                "input": "~/.cache/elvesplant/in_buff",
                "output": "~/.cache/elvesplant/out_buff",
            })

        self.var_opt = self.info["var_opt"]
        self.num_threads = self.var_opt.get("num_threads", 1)
        corpus_len = len(self.corpus)
        if corpus_len % self.num_threads != 0:
            self.extra_split = True
            self.num_threads += 1
        queue_size = self.var_opt.get("queue_size", 1)
        self.read_queues = [
            multi.Queue(maxsize=queue_size)
            for i in range(self.num_threads)]
        self.writ_queues = [
            multi.Queue(maxsize=queue_size)
            for i in range(self.num_threads)]

        self.proced_info = {}
        for proced_name in self.var_opt["to_do"]:
            self.proced_info[proced_name] = {}
            if proced_name == "tokenization":
                self.proced_info[proced_name] = {
                    "English script": MosesTokenizer("en"),
                    "German script": MosesTokenizer("de"),
                }
            elif proced_name == "subword segmentation":
                sys_path = self.var_opt["sentencepiece"]["sys_path"]
                subword_system = SentencePieceProcessor(model_file=sys_path)
                self.proced_info[proced_name]["system"] = subword_system
            elif proced_name == "length control":
                self.proced_info[proced_name]["max"] = self.var_opt["max_len"]
            elif proced_name == "draw out samples":
                self.proced_info[proced_name]["size"] = (
                    self.var_opt["sample_size"])

        buff_dir_list = []
        for buff_name in ["input", "output"]:
            buff_dir_path = os.path.abspath(self.buff_dir[buff_name])
            os.makedirs(buff_dir_path, exist_ok=True)
            buff_dir_list.append(buff_dir_path)
        self.buff_dir_list = buff_dir_list
        self.split_path = {}

        for data_name, detail_info in self.corpus.info.items():
            for buff_dir_path in buff_dir_list:
                os.makedirs(f"{buff_dir_path}/{data_name}", exist_ok=True)
            if self.extra_split:
                num_threads = self.num_threads - 1
            else:
                num_threads = self.num_threads
            split_size = len(self.corpus) // num_threads
            in_file_path = detail_info["file_path"]
            in_file_dir = f"{buff_dir_list[0]}/{data_name}"
            comm_str = (
                f"split -l {split_size} "
                f"{in_file_path} {in_file_dir}/")
            shell_proc = subprocess.run(
                comm_str.split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True, text=True)
            print(shell_proc.stderr)
            file_list = os.listdir(in_file_dir)
            self.split_path[data_name] = file_list

        self.out_buff = []
        for i in range(len(file_list)):
            out_file_dict = {}
            for data_name in self.corpus.info:
                out_file_name = os.path.basename(
                    self.split_path[data_name][i])
                out_path = (
                    f"{buff_dir_list[1]}/"
                    f"{data_name}/{out_file_name}")
                out_file_dict[data_name] = open(
                    out_path, "w", encoding="utf-8")
            self.out_buff.append(out_file_dict)

    def run(self):
        """Read and write."""
        writ_processes = []
        for i in range(self.num_threads):

            read_corpus = self.corpus
            if self.buff_dir is not None:
                rev_corp_info = deepcopy(self.corpus.info)
                for data_name, detail_info in rev_corp_info.items():
                    in_buff_path = f"{self.buff_dir_list[0]}/{data_name}"
                    file_name = self.split_path[data_name][i]
                    detail_info["file_path"] = f"{in_buff_path}/{file_name}"
                read_corpus = Corpus(self.corpus.name, rev_corp_info)
            read_proc = multi.Process(
                target=read_through,
                args=(read_corpus, self.read_queues[i]))
            read_proc.start()

            writ_proc = multi.Process(
                target=record,
                args=(
                    self.writ_queues[i],
                    self.out_buff[i],
                    self.proced_info,
                ),
            )
            writ_proc.start()
            writ_processes.append(writ_proc)

            work_proc = multi.Process(
                target=process_corpus,
                args=(
                    self.read_queues[i],
                    self.writ_queues[i],
                    self.corpus.info,
                    self.proced_info,
                ),
            )
            work_proc.start()

        for writ_proc in writ_processes:
            writ_proc.join()
        for read_que in self.read_queues:
            read_que.close()
        for writ_que in self.writ_queues:
            writ_que.close()
        for out_file_dict in self.out_buff:
            for out_file in out_file_dict.values():
                out_file.close()
