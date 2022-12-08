"""Classes and utlitiles for n-tuple corpora."""


class Corpus:
    """A basic n-tuple corpus.

    This corpus binds multiple sorts of data to a single iterator.
    It is assumed that at least one part of this corpus is text.
    """

    def __init__(self, name: str, corpus_info: dict):
        self.name = name
        self.info = corpus_info
        self.file_readers = {}
        self.curr_line_index = 0
        self.load()

    def __del__(self):
        for file_reader in self.file_readers.values():
            file_reader.close()

    def __iter__(self):
        return self

    def __len__(self):
        file_reader = None
        for detail_info in self.info.values():
            if detail_info["data_type"] == "text":
                file_reader = open(detail_info["file_path"], mode="rb")
                break
        if file_reader is None:
            raise NotImplementedError
        else:
            count = 0
            for line in file_reader:
                count += 1
            file_reader.close()
            return count

    def __next__(self):
        curr_line = {}
        for data_name, file_reader in self.file_readers.items():
            try:
                curr_line[data_name] = next(file_reader)
            except StopIteration as stop:
                self.curr_line_index = -1
                raise StopIteration from stop
        self.curr_line_index += 1
        return self.decode(curr_line)

    def load(self):
        """Load data from their sources."""
        for data_name, detail_info in self.info.items():
            if detail_info["data_type"] == "text":
                self.file_readers[data_name] = open(
                    detail_info["file_path"], mode="rb")
            else:
                raise NotImplementedError

    def decode(self, curr_line: dict):
        """Process given data."""
        output = {}
        for i, (data_name, line_content) in enumerate(curr_line.items()):
            if line_content is not None:
                detail_info = self.info[data_name]
                transmit_arg = {}
                if detail_info["data_type"] == "text":
                    transmit_arg["encoding"] = detail_info["encoding"]
                    file_type = detail_info.get("file_type", "txt")
                    transmit_arg["file_type"] = file_type
                    transmit_arg["data_index"] = i
                    output[data_name] = self.text_decoder(
                        line_content, **transmit_arg)
                else:
                    raise NotImplementedError
            else:
                output[data_name] = line_content
        output["line_index"] = self.curr_line_index
        return output

    @staticmethod
    def text_decoder(
        byte_data: bytes, encoding: str, file_type="txt",
        data_index: int = None,
    ) -> str:
        """Process a datum that is assumed to be text."""
        text_data = byte_data.decode(encoding)
        seq = text_data.rstrip("\n")
        if file_type == "tsv":
            bitext = seq.split("\t")
            return bitext[data_index]
        return seq
