import nltk
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

from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
)
from torchvision.transforms import (
    Compose,
    ToTensor,
)

from api import Token


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
        for item in tqdm.tqdm(self.df.itertuples(), total=len(self.df)):
            for sentence in self.sent_tokenizer.tokenize(item.text):
                yield (
                    [Token.bos] +
                    self.word_tokenizer.tokenize(sentence) +
                    [Token.eos]
                )

    '''
    def _wordmap_path(self, field):
        return os.path.join(os.getenv('CACHE_DIR'), f'wordmap-field-{field}.csv')

    def _make_wordmap(self, field):
        counter = {}
        for sentence in self._iterate_sentences():
            for word in sentence:
                counter[word] = counter.get(word, 0) + 1

        df = pd.DataFrame(list(counter.items()), columns=['word', 'word_count'])
        df = df.set_index('word').sort_values('word_count', ascending=False)
        df.to_csv(self._wordmap_path(field=field))
    '''

    def _word_embedding_path(self, field):
        return os.path.join(os.getenv('CACHE_DIR'), f'word-embedding-field-{field}.pkl')

    def _make_word_embedding(self, field):
        word2vec = Word2Vec(self, size=self.embedding_size, min_count=self.min_word_freq, workers=24)
        word2vec.wv.save(self._word_embedding_path(field=field))

    def __init__(self,
                 df,
                 field,
                 min_word_freq,
                 max_report_length,
                 max_sentence_length,
                 embedding_size,
                 is_train):

        self.df = df
        self.min_word_freq = min_word_freq
        self.max_report_length = max_report_length
        self.max_sentence_length = max_sentence_length
        self.embedding_size = embedding_size
        self.is_train = is_train

        self.sent_tokenizer = PunktSentenceTokenizer()
        self.word_tokenizer = TweetTokenizer()

        self.num_view_position = max(self.view_position_to_index.values()) + 1

        if is_train:
            '''
            if not os.path.isfile(self._wordmap_path(field=field)):
                self._make_wordmap(field=field)
            '''

            if not os.path.isfile(self._word_embedding_path(field=field)):
                self._make_word_embedding(field=field)

        word_vectors = KeyedVectors.load(self._word_embedding_path(field=field))

        self.index_to_word = word_vectors.index2entity + [Token.unk, Token.pad]
        self.word_to_index = dict(zip(self.index_to_word, range(len(self.index_to_word))))
        self.word_embedding = np.concatenate([
            word_vectors.vectors,
            np.zeros((2, embedding_size)),
        ], axis=0).astype(np.float32)

        '''
        df_word = pd.read_csv(self._wordmap_path(field=field), index_col='word')
        sel = (df_word.word_count >= min_word_freq)
        df_word = df_word[sel]

        df_word.loc[Token.eos] = max(df_word.word_count) + 1
        df_word.loc[Token.unk] = sum(~sel)
        df_word.loc[Token.bos] = 0
        df_word.loc[Token.pad] = -1

        df_word = df_word.sort_values('word_count', ascending=False)
        self.word_to_index = dict(zip(df_word.index, range(len(df_word))))
        self.index_to_word = df_word.index
        '''

        # TODO(stmharry): ColorJitter
        self.transform = Compose([
            ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]

        path = os.path.join(os.getenv('CACHE_DIR'), 'images', f'{item.dicom_id}.png')
        image = PIL.Image.open(path)
        image = self.transform(image)

        view_position_index = self.view_position_to_index.get(item.view_position, 0)
        view_position = torch.arange(self.num_view_position) == view_position_index
        view_position = torch.as_tensor(view_position, dtype=torch.float)

        _item = {
            'image': image,
            'view_position': view_position,
        }

        if self.is_train:
            text = []
            sent_length = []

            sentences = self.sent_tokenizer.tokenize(item.text)
            sentences = [''] + sentences[:min(len(sentences), self.max_report_length)]
            for sentence in sentences:
                words = self.word_tokenizer.tokenize(sentence)

                num_words = min(len(words), self.max_sentence_length)
                words = words[:num_words]

                words = torch.as_tensor((
                    [self.word_to_index[Token.bos]] +
                    [self.word_to_index.get(word, self.word_to_index[Token.unk]) for word in words] +
                    [self.word_to_index[Token.eos]] +
                    [self.word_to_index[Token.pad]] * (self.max_sentence_length - num_words)
                ), dtype=torch.long)

                text.append(words)
                sent_length.append(num_words + 2)

            text = torch.stack(text, 0)
            sent_length = torch.as_tensor(sent_length, dtype=torch.long)
            text_length = torch.as_tensor(sent_length.numel(), dtype=torch.long)

            # TODO(stmharry): really load label
            label = torch.ones((text_length, 16), dtype=torch.float)

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
