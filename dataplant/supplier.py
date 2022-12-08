"""Classes and utlitiles for data supply routines"""
from ape_filter import APEFilter


class DataSupplier:
    """A basic data supply framework."""

    pre_routines = {
        "ape_filter": APEFilter,
    }

    def __init__(self, corpora, supp_info):
        self.corpora = corpora
        self.info = supp_info
        self.pre_plan = {}
        if "pretreat" in self.info.keys():
            pre_info = self.info["pretreat"]
            corpus_name_list = [corpus.name for corpus in corpora]
            for treat_name, detail_info in pre_info.items():
                self.pre_plan[treat_name] = {}
                for corpus_name in detail_info["corpora"]:
                    corpus = self.corpora[corpus_name_list.index(corpus_name)]
                    self.pre_plan[treat_name][corpus_name] = (
                        self.pre_routines[treat_name](corpus, self.info))

    def run(self):
        """currently, perform pretreatments only."""
        for detail_info in self.pre_plan.values():
            for treat_oper in detail_info.values():
                treat_oper.run()
