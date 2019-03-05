import functools

from pathlib import Path

from chexpert_labeler.loader import Loader
from chexpert_labeler.stages import Extractor, Classifier, Aggregator
from chexpert_labeler.constants import CATEGORIES

working_dir = Path(__file__).parent

Loader = functools.partial(
    Loader,
    extract_impression=False,
)
Extractor = functools.partial(
    Extractor,
    mention_phrases_dir=working_dir.joinpath('phrases', 'mention'),
    unmention_phrases_dir=working_dir.joinpath('phrases', 'unmention'),
)
Classifier = functools.partial(
    Classifier,
    pre_negation_uncertainty_path=working_dir.joinpath('patterns', 'pre_negation_uncertainty.txt'),
    negation_path=working_dir.joinpath('patterns', 'negation.txt'),
    post_negation_uncertainty_path=working_dir.joinpath('patterns', 'post_negation_uncertainty.txt'),
)
Aggregator = functools.partial(
    Aggregator,
    CATEGORIES,
)
