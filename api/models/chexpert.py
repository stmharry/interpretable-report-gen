import io

import chexpert


class CheXpert(object):
    def __init__(self):
        self.initialized = False

    def __call__(self, s):
        if not self.initialized:
            self.extractor = chexpert.Extractor()
            self.classifier = chexpert.Classifier()
            self.aggregator = chexpert.Aggregator()
            self.initialized = True

        s = '\n'.join(s)
        f = io.BytesIO(s.encode())
        loader = chexpert.Loader(reports_path=f)
        loader.load()

        self.extractor.extract(loader.collection)
        self.classifier.classify(loader.collection)
        labels = self.aggregator.aggregate(loader.collection)

        return labels
