import numpy as np
import os
import pandas as pd
import PIL.Image
import torch
import torch.utils.data
import tqdm

from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.casual import TweetTokenizer

from torchvision.transforms import Lambda, Resize, Compose, ColorJitter, ToTensor, Normalize

from api import Phase, Token
from api.metrics import CiderScorer
from api.utils import to_numpy
from api.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MimicCXRDataset(torch.utils.data.Dataset):
    view_position_to_index = {
        'AP': 1,
        'PA': 2,
        'LL': 3,
        'LATERAL': 3,
    }

    def __iter__(self):
        yield from self._iterate_sentences()

    def _iterate_sentences(self):
        for item in tqdm.tqdm(self.df_sentence_label.itertuples(), total=len(self.df_sentence_label)):
            yield (
                [Token.bos] +
                self.df_sentence_label['sentence'].split()
                [Token.eos]
            )

    def _sentence_label_path(self):
        postfix = '' if self.debug_use_dataset is None else f'.{self.debug_use_dataset}'
        return os.path.join(os.getenv('CACHE_DIR'), f'sentence-label-field-{self.field}{postfix}.tsv')

    def _make_sentence_label(self):
        def _tokenize_join(sentence):
            return ' '.join(self.word_tokenizer.tokenize(sentence))

        df = self.df.drop_duplicates('rad_id')[['rad_id', 'text']]

        df_regex_path = os.path.join(os.getenv('CACHE_DIR'), 'labels.tsv')
        df_regex = pd.read_csv(df_regex_path, delimiter='\t')
        regexes = df_regex.groupby('label', sort=False)['regex'].aggregate('|'.join)

        _dfs = []
        for item in tqdm.tqdm(df.itertuples(), total=len(df)):
            sentences = self.sent_tokenizer.tokenize(item.text)
            sentences = sentences[:min(len(sentences), self.max_report_length)]
            sentences = pd.Series(sentences, name='sentence')

            _series = []
            _series.append(pd.Series([item.rad_id] * len(sentences), name='rad_id'))

            for label in regexes.index:
                _series.append(pd.Series(
                    1 - sentences.str.extract(r'(\b)({})(\b)'.format(regexes[label])).isnull().all(axis=1).astype(np.int),
                    name=f'label_{label}',
                ))

            sentences = sentences.apply(_tokenize_join)
            _series.append(sentences)

            _dfs.append(pd.concat(_series, axis=1))

        _df = pd.concat(_dfs)
        _df.to_csv(self._sentence_label_path(), index=False, sep='\t')

    def _word_embedding_path(self):
        return os.path.join(os.getenv('CACHE_DIR'), f'word-embedding-field-{self.field}.pkl')

    def _make_word_embedding(self):
        word2vec = Word2Vec(self, size=self.embedding_size, min_count=self.min_word_freq, workers=24)
        word2vec.wv.save(self._word_embedding_path())

    def _cider_cache_path(self):
        return os.path.join(os.getenv('CACHE_DIR'), f'cider-cache.pkl')

    def _make_cider_cache(self):
        cider_scorer = CiderScorer()
        for sentence in tqdm.tqdm(self.df_sentence_label['sentence']):
            cider_scorer += (None, [sentence])
        cider_scorer.compute_doc_freq()
        torch.save({
            'document_frequency': cider_scorer.document_frequency,
            'ref_len': np.log(float(len(cider_scorer.crefs))),
        }, self._cider_cache_path())

    def __init__(self,
                 df,
                 field,
                 min_word_freq,
                 max_report_length,
                 max_sentence_length,
                 embedding_size,
                 phase,
                 kwargs):

        self.df = df
        self.field = field
        self.min_word_freq = min_word_freq
        self.max_report_length = max_report_length
        self.max_sentence_length = max_sentence_length
        self.embedding_size = embedding_size
        self.phase = phase

        self.debug_use_dataset  = kwargs['debug_use_dataset']
        self.debug_one_sentence = kwargs['debug_one_sentence']
        self.__use_densenet     = kwargs['__use_densenet']

        self.sent_tokenizer = PunktSentenceTokenizer()
        self.word_tokenizer = TweetTokenizer()

        self.view_position_size = max(self.view_position_to_index.values()) + 1

        # df_sentence_label

        if (phase is None) and (not os.path.isfile(self._sentence_label_path())):
            self._make_sentence_label()

        self.df_sentence_label = pd.read_csv(self._sentence_label_path(), sep='\t', dtype={'rad_id': str}).set_index('rad_id')
        self.label_columns = [column for column in self.df_sentence_label.columns if column.startswith('label_')]
        self.label_size = len(self.label_columns)

        if self.debug_use_dataset:
            self.df = self.df[self.df.rad_id.isin(self.df_sentence_label.index.unique())]

        if self.debug_one_sentence:
            self.max_report_length = 1

        # df_cache

        if (phase is None) and (not os.path.isfile(self._cider_cache_path())):
            self._make_cider_cache()

        if phase == Phase.train:
            self.df_cache = torch.load(self._cider_cache_path())

        # word_embedding

        if (phase == Phase.train) and (not os.path.isfile(self._word_embedding_path())):
            self._make_word_embedding()

        word_vectors = KeyedVectors.load(self._word_embedding_path())
        self.index_to_word = np.array(word_vectors.index2entity + [Token.unk])
        self.word_to_index = dict(zip(self.index_to_word, range(len(self.index_to_word))))
        self.word_embedding = np.concatenate([
            word_vectors.vectors,
            np.zeros((1, embedding_size)),
        ], axis=0).astype(np.float32)

        # transform

        if phase == Phase.train:
            jitter = [ColorJitter(brightness=0.5, contrast=0.5)]
        else:
            jitter = []

        if self.__use_densenet:
            self.transform = Compose((
                [Lambda(lambda img: img.convert('RGB'))] +
                [Resize(256)] +
                jitter +
                [ToTensor()] +
                [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ))
        else:
            self.transform = Compose((
                [Resize(256)] +
                jitter +
                [ToTensor()]
            ))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        df_rad = self.df.loc[self.df.rad_id == item.rad_id]

        path = os.path.join(os.getenv('CACHE_DIR'), 'images', f'{item.dicom_id}.png')
        image = PIL.Image.open(path)
        image = self.transform(image)

        view_position_index = self.view_position_to_index.get(item.view_position, 0)
        view_position_indices = [self.view_position_to_index.get(view_position, 0) for view_position in df_rad.view_position]

        view_position = torch.cat([
            torch.arange(self.view_position_size) == view_position_index,
            (torch.arange(self.view_position_size).unsqueeze(1) == torch.as_tensor(view_position_indices)).any(1),
        ], 0)
        view_position = torch.as_tensor(view_position, dtype=torch.float)

        _item = {
            'item_index': index,
            'image': image,
            'view_position': view_position,
        }

        if (self.phase == Phase.train) or (self.phase == Phase.val):
            text = []
            sent_length = []

            df_sentence = self.df_sentence_label.loc[[item.rad_id]]
            for item_sentence in df_sentence.itertuples():
                words = item_sentence.sentence.split()

                num_words = min(len(words), self.max_sentence_length - 1) + 1
                words = torch.as_tensor((
                    [self.word_to_index.get(word, self.word_to_index[Token.unk]) for word in words[:num_words - 1]] +
                    [self.word_to_index[Token.eos]] +
                    [0] * (self.max_sentence_length - num_words)
                ), dtype=torch.long)

                text.append(words)
                sent_length.append(num_words)

            text = torch.stack(text, 0)
            sent_length = torch.as_tensor(sent_length, dtype=torch.long)
            text_length = torch.as_tensor(sent_length.numel(), dtype=torch.long)

            label = torch.as_tensor(df_sentence[self.label_columns].values, dtype=torch.float)

            num = torch.arange(text_length, dtype=torch.long).unsqueeze(1)
            stop = torch.as_tensor(num == text_length - 1, dtype=torch.float)

            _item.update({
                'text_length': text_length,
                'text': text,
                'label': label,
                'stop': stop,
                'sent_length': sent_length,
            })

        return _item

    def convert_sentence(self, sent, sent_length, text_length):
        word = pack_padded_sequence(sent, length=sent_length)
        length = pad_packed_sequence(sent_length, length=text_length).sum(1)

        word = self.index_to_word[to_numpy(word)]
        words = np.split(word, np.cumsum(to_numpy(length)))[:-1]

        return np.array([' '.join(word) for word in words])
