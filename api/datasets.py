import numpy as np
import os
import PIL.Image
import torch
import torch.utils.data
import tqdm
import re
import string
import pandas as pd

from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.casual import TweetTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

from torchvision.transforms import Compose, ToTensor

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
    def _sentence_labels_regex_path(self):
        return os.path.join(os.getenv('CACHE_DIR'), 'Sentence Labeling Categories - Categories.csv')

    def _labels_path(self, field):
        return os.path.join(os.getenv('CACHE_DIR'), f'sentence-labels-{field}.pt')

    def _labels_map_path(self, field):
        return os.path.join(os.getenv('CACHE_DIR'), f'sentence-labels-map-{field}.pt')

    def _read_categories_df(self):
        return pd.read_csv(self._sentence_labels_regex_path(),header=[0, 1])

    def _sentence_labeler(self, s, sentence_categories_df):
        punct_to_remove = ''.join([x for x in string.punctuation if x != '-'])
        s = s.translate(punct_to_remove).lower()
        categories = []
        for col in sentence_categories_df.columns.tolist():
            regex_set = set(['(^|\s)%s($|\s)' % x for x in sentence_categories_df[col] if type(x) is str])
            for r in regex_set:
                if len(re.findall(r, s)) > 0:
                    categories.append(col)
                    break
        return categories

    def _make_labels(self, field):
        sentence_categories_df = self._read_categories_df()
        mlb = self._fit_binarizer(sentence_categories_df)
        labels = []
        for index in range(len(self.df)):
            item = self.df.iloc[index]
            category = []

            sentences = self.sent_tokenizer.tokenize(item.text)
            sentences = [''] + sentences[:min(len(sentences), self.max_report_length)]
            for sentence in sentences:
                words = self.word_tokenizer.tokenize(sentence)

                num_words = min(len(words), self.max_sentence_length)
                words = words[:num_words]

                new_sentence = ' '.join(words)

                categories = self._sentence_labeler(new_sentence, sentence_categories_df)
                st = set()
                for (lv1, lv2) in categories:
                    st.add(lv2)
                cat_vec = mlb.transform([st]).flatten()
                category.append(cat_vec)

            labels.append(torch.as_tensor(category, dtype=torch.float))
        torch.save(labels, self._labels_path(field=field))
        torch.save(mlb.classes_, self._labels_map_path(field=field))

    def _fit_binarizer(self, sentence_categories_df):
        '''
        transform categories into 1D vec and prepare for regex
        '''
        cols = sentence_categories_df.columns.tolist()
        mlb = MultiLabelBinarizer()
        cols_fixed = []
        st = set()
        last_lvl1 = None
        for (lvl1, lvl2) in cols:
            if lvl1.startswith('Unnamed'):
                st.add(lvl2)
                cols_fixed.append((last_lvl1, lvl2))
            else:
                st.add(lvl1)
                st.add(lvl2)
                cols_fixed.append((lvl1, lvl2))
                last_lvl1 = lvl1
        lst = [st]
        mlb.fit_transform(lst)
        sentence_categories_df.columns = pd.MultiIndex.from_tuples(cols_fixed)
        return mlb

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
            if not os.path.isfile(self._labels_path(field=field)):
                self._make_labels(field=field)

        word_vectors = KeyedVectors.load(self._word_embedding_path(field=field))
        self.labels = torch.load(self._labels_path(field=field))


        self.index_to_word = word_vectors.index2entity + [Token.unk]
        self.word_to_index = dict(zip(self.index_to_word, range(len(self.index_to_word))))
        self.word_embedding = np.concatenate([
            word_vectors.vectors,
            np.zeros((1, embedding_size)),
        ], axis=0).astype(np.float32)

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
            'item_index': index,
            'image': image,
            'view_position': view_position,
        }

        if self.is_train:
            text = []
            sent_length = []

            sentences = self.sent_tokenizer.tokenize(item.text)
            sentences = sentences[:min(len(sentences), self.max_report_length)]
            for sentence in sentences:
                words = self.word_tokenizer.tokenize(sentence)

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

            label = self.labels[index]

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
