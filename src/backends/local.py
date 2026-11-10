from src.training import Paths, RunOptions, TrainConfig, train


class LocalBackend:
    name = "local"

    def __init__(self, paths: Paths, ensure_corpus=None):
        self.paths = paths
        self._ensure_corpus = ensure_corpus

    def run(self, settings: TrainConfig, options: RunOptions) -> str | None:
        if self._ensure_corpus is not None:
            self._ensure_corpus(self.paths.corpus_file)
        return train(settings, self.paths, options)
