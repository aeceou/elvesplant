"""Classes and utlitiles for data processing routines"""
import os
import weakref

from omegaconf import DictConfig
import hydra

from corpus import Corpus
from supplier import DataSupplier


ELVES_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))


class DataPlant:
    """A basic data processor."""

    def __init__(self, plant_info: DictConfig):
        self.info = plant_info
        self.supp_info = self.info["supplier"]
        self.corpora = []
        for corpus_name, corpus_info in self.supp_info["corpora"].items():
            self.corpora.append(Corpus(corpus_name, corpus_info))
        self.supplier = DataSupplier(self.corpora, self.supp_info)

    def __del__(self):
        for i in range(len(self.corpora)):
            del self.corpora[i]

    def run(self):
        """Process data in any way."""
        self.supplier.run()


@hydra.main(
    version_base=None,
    config_path=f"{ELVES_DIR}/conf",
    config_name="plant")
def run_with_hydra(conf: DictConfig) -> None:
    """Process data by using Hydra."""
    data_plant = DataPlant(conf)
    data_plant.run()
    del data_plant


if __name__ == "__main__":
    run_with_hydra()
