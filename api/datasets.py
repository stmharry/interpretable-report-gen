import glob
import lxml
import natsort
import numpy as np
import os
import pandas as pd
import PIL.Image
import pydicom
import torch
import torch.utils.data
import tqdm

from gensim.models import Word2Vec, KeyedVectors
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.casual import TweetTokenizer

from torchvision.transforms import Lambda, Resize, Compose, ColorJitter, ToTensor, Normalize

from chexpert_labeler import CATEGORIES

from api import Mode, Phase, Token
from api.models.base import DataParallelCPU
from api.models.nondiff import CheXpert
from api.metrics.nlp import CiderScorer


class Dataset(torch.utils.data.Dataset):
    view_position_to_index = {
        'AP': 1,
        'PA': 2,
        'LL': 3,
        'LATERAL': 3,
    }
    view_position_size = max(view_position_to_index.values()) + 1

    df_cache = None
    index_to_word = None

    def __iter__(self):
        yield from self._iterate_sentences()

    def _iterate_sentences(self):
        for item in tqdm.tqdm(self.df_sentence.itertuples(), total=len(self.df_sentence)):
            yield (
                [Token.bos] +
                item.sentence.split() +
                [Token.eos]
            )

    def _make_cider_cache(self):
        cider_scorer = CiderScorer()
        for sentence in tqdm.tqdm(self.df_sentence.sentence):
            cider_scorer += (None, [sentence])
        cider_scorer.compute_doc_freq()
        torch.save({
            'document_frequency': cider_scorer.document_frequency,
            'ref_len': np.log(float(len(cider_scorer.crefs))),
        }, self._cider_cache_path())

    def _make_word_embedding(self):
        word2vec = Word2Vec(self, size=self.embedding_size, min_count=self.min_word_freq, workers=24)
        word2vec.wv.save(self._word_embedding_path())

    def __init__(self, phase, kwargs):
        self.mode        = Mode[kwargs['mode']]
        self.image_size  = kwargs['image_size']
        self.hidden_size = kwargs['hidden_size']

        self.debug_use_dataset  = kwargs['debug_use_dataset']
        self.debug_one_sentence = kwargs['debug_one_sentence']
        self.__use_densenet     = kwargs['__use_densenet']

        self.sent_tokenizer = PunktSentenceTokenizer()
        self.word_tokenizer = TweetTokenizer()

        if phase == Phase.train:
            jitter = [ColorJitter(brightness=0.5, contrast=0.5)]
        else:
            jitter = []

        if self.__use_densenet:
            self.transform = Compose((
                [Lambda(lambda img: img.convert('RGB'))] +
                [Resize((256, 256))] +
                jitter +
                [ToTensor()] +
                [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ))
        else:
            self.transform = Compose((
                [Resize((256, 256))] +
                jitter +
                [ToTensor()]
            ))


class MimicCXRDataset(Dataset):
    df_sentence = None
    df_chexpert_label = None

    def _report_chexpert_path(self):
        postfix = '' if self.debug_use_dataset is None else f'.{self.debug_use_dataset}'
        return os.path.join(os.getenv('CACHE_DIR'), f'report-chexpert-field-{self.field}{postfix}.tsv')

    def _make_report_chexpert(self):
        df = self.df.drop_duplicates('rad_id')[['rad_id', 'text']]

        chexpert = DataParallelCPU(CheXpert, verbose=True)
        chexpert_label = chexpert(df.text.values)

        _df = pd.DataFrame({'rad_id': df.rad_id})
        for (num, category) in enumerate(CATEGORIES):
            _df[category] = chexpert_label[:, num]

        _df.to_csv(self._report_chexpert_path(), sep='\t', index=False)

    def _sentence_path(self):
        postfix = '' if self.debug_use_dataset is None else f'.{self.debug_use_dataset}'
        return os.path.join(os.getenv('CACHE_DIR'), f'sentence-label-field-{self.field}{postfix}.tsv')

    def _make_sentence(self):
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
        _df.to_csv(self._sentence_path(), index=False, sep='\t')

    def _word_embedding_path(self):
        return os.path.join(os.getenv('CACHE_DIR'), f'word-embedding-field-{self.field}.pkl')

    def _cider_cache_path(self):
        return os.path.join(os.getenv('CACHE_DIR'), f'cider-cache.pkl')

    def __init__(self,
                 df,
                 field,
                 min_word_freq,
                 max_report_length,
                 max_sentence_length,
                 embedding_size,
                 phase,
                 kwargs):

        super(MimicCXRDataset, self).__init__(phase, kwargs)

        self.df = df
        self.field = field
        self.min_word_freq = min_word_freq
        self.max_report_length = max_report_length
        self.max_sentence_length = max_sentence_length
        self.embedding_size = embedding_size
        self.phase = phase

        if (phase is None) and (not os.path.isfile(self._report_chexpert_path())):
            self._make_report_chexpert()

        if MimicCXRDataset.df_chexpert_label is None:
            MimicCXRDataset.df_chexpert_label = pd.read_csv(self._report_chexpert_path(), sep='\t', dtype={'rad_id': str}).set_index('rad_id')

        if (phase is None) and (not os.path.isfile(self._sentence_path())):
            self._make_sentence()

        if MimicCXRDataset.df_sentence is None:
            MimicCXRDataset.df_sentence = pd.read_csv(self._sentence_path(), sep='\t', dtype={'rad_id': str}).set_index('rad_id')
            MimicCXRDataset.label_columns = [column for column in self.df_sentence.columns if column.startswith('label_')]
            MimicCXRDataset.label_size = len(self.label_columns)

        if self.debug_use_dataset:
            self.df = self.df[self.df.rad_id.isin(MimicCXRDataset.df_sentence.index.unique())]

        if self.debug_one_sentence:
            self.max_report_length = 1

        if (phase is None) and (not os.path.isfile(self._cider_cache_path())):
            self._make_cider_cache()

        if MimicCXRDataset.df_cache is None:
            MimicCXRDataset.df_cache = torch.load(self._cider_cache_path())

        if (phase == Phase.train) and (not os.path.isfile(self._word_embedding_path())):
            self._make_word_embedding()

        if MimicCXRDataset.index_to_word is None:
            word_vectors = KeyedVectors.load(self._word_embedding_path())
            MimicCXRDataset.index_to_word = np.array(word_vectors.index2entity + [Token.unk])
            MimicCXRDataset.word_to_index = dict(zip(self.index_to_word, range(len(self.index_to_word))))
            MimicCXRDataset.word_embedding = np.concatenate([
                word_vectors.vectors,
                np.zeros((1, embedding_size)),
            ], axis=0).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        df_rad = self.df.loc[self.df.rad_id == item.rad_id]

        _item = {
            'item_index': index,
        }

        if self.mode & Mode.enc_image:
            path = os.path.join(os.getenv('CACHE_DIR'), 'images', f'{item.dicom_id}.png')
            image = PIL.Image.open(path)
            image = self.transform(image)

            view_position_index = MimicCXRDataset.view_position_to_index.get(item.view_position, 0)
            view_position_indices = [MimicCXRDataset.view_position_to_index.get(view_position, 0) for view_position in df_rad.view_position]

            view_position = torch.cat([
                torch.arange(MimicCXRDataset.view_position_size) == view_position_index,
                (torch.arange(MimicCXRDataset.view_position_size).unsqueeze(1) == torch.as_tensor(view_position_indices)).any(1),
            ], 0)
            view_position = torch.as_tensor(view_position, dtype=torch.float)

            _item.update({
                'image': image,
                'view_position': view_position,
            })
        else:
            _item.update({
                'image': torch.randn((self.image_size * self.image_size, self.hidden_size), dtype=torch.float),
                'view_position': torch.randint(0, 2, (2 * MimicCXRDataset.view_position_size,), dtype=torch.float),
            })

        text = []
        sent_length = []

        df_sentence = MimicCXRDataset.df_sentence.loc[[item.rad_id]]
        if self.mode & Mode.as_one_sentence:
            sentences = [' '.join(df_sentence.sentence)]
        else:
            sentences = df_sentence.sentence

        for sentence in sentences:
            words = sentence.split()

            num_words = min(len(words), self.max_sentence_length - 1) + 1
            words = torch.as_tensor((
                [MimicCXRDataset.word_to_index.get(word, MimicCXRDataset.word_to_index[Token.unk]) for word in words[:num_words - 1]] +
                [MimicCXRDataset.word_to_index[Token.eos]] +
                [0] * (self.max_sentence_length - num_words)
            ), dtype=torch.long)

            text.append(words)
            sent_length.append(num_words)

        text = torch.stack(text, 0)
        sent_length = torch.as_tensor(sent_length, dtype=torch.long)
        text_length = torch.as_tensor(sent_length.numel(), dtype=torch.long)

        # TODO(stmharry): remove label as it is not providing any benefit
        label = torch.as_tensor(df_sentence[MimicCXRDataset.label_columns].values, dtype=torch.float)

        df_chexpert_label = MimicCXRDataset.df_chexpert_label.loc[item.rad_id]
        chexpert_label = torch.as_tensor(df_chexpert_label[CATEGORIES].values, dtype=torch.long)

        num = torch.arange(text_length, dtype=torch.long).unsqueeze(1)
        stop = torch.as_tensor(num == text_length - 1, dtype=torch.float)

        _item.update({
            'text_length': text_length,
            'text': text,
            'label': label,
            'chexpert_label': chexpert_label,
            'stop': stop,
            'sent_length': sent_length,
        })

        return _item


class OpenIDataset(Dataset):
    df = None
    df_meta = None
    df_report = None
    df_sentence = None

    def _meta_path(self):
        return os.path.join(os.getenv('CACHE_DIR'), 'open-i', f'meta.tsv')

    def _make_meta(self):
        paths = glob.glob(os.path.join(os.getenv('OPENI_DIR'), 'ecgen-radiology', '*.xml'))

        df = []
        for path in tqdm.tqdm(natsort.natsorted(paths)):
            rad_id = os.path.splitext(os.path.basename(path))[0]

            tree = lxml.etree.parse(path)

            images = tree.findall('.//parentImage')
            for image in images:
                dicom_id = image.get('id')
                dicom_postfix = dicom_id.split('_')[-1]

                dcm_path = os.path.join(os.getenv('OPENI_DIR'), 'dcm', rad_id, f'{rad_id}_{dicom_postfix}.dcm')

                try:
                    dcm = pydicom.read_file(dcm_path, stop_before_pixels=True)
                    view_position = dcm.get((0x0018, 0x5101)).value
                except Exception:
                    pass
                else:
                    df.append({
                        'rad_id': rad_id,
                        'dicom_id': dicom_id,
                        'view_position': view_position,
                    })

        df = pd.DataFrame(df)
        df.to_csv(self._meta_path(), sep='\t', index=False)

    def _report_chexpert_path(self):
        return os.path.join(os.getenv('CACHE_DIR'), 'open-i', f'report-chexpert-field-{self.field}.tsv')

    def _make_report_chexpert(self):
        paths = glob.glob(os.path.join(os.getenv('OPENI_DIR'), 'ecgen-radiology', '*.xml'))

        for path in tqdm.tqdm(natsort.natsorted(paths)):
            rad_id = os.path.splitext(os.path.basename(path))[0]
            tree = lxml.etree.parse(path)

            try:
                text = tree.find(f'.//AbstractText[@Label="{self.field.upper()}"]').text.lower()
            except Exception:
                pass
            else:
                df_report.append({
                    'rad_id': rad_id,
                    'text': text,
                })

        df_report = pd.DataFrame(df_report)

        chexpert = DataParallelCPU(CheXpert, num_jobs=None, maxtasksperchild=256, verbose=True)
        chexpert_label = chexpert(df_report.text.values)
        for (num, category) in enumerate(CATEGORIES):
            df_report[category] = chexpert_label[:, num]

        df_report = df_report[['rad_id'] + CATEGORIES + ['text']]
        df_report.to_csv(self._report_chexpert_path(), sep='\t', index=False)

    def _sentence_path(self):
        return os.path.join(os.getenv('CACHE_DIR'), 'open-i', f'sentence-field-{self.field}.tsv')

    def _make_sentence(self):
        df_sentence = []
        for item in self.df_report.itertuples():
            sentences = self.sent_tokenizer.tokenize(item.text)
            for sentence in sentences:
                sentence = ' '.join(self.word_tokenizer.tokenize(sentence))

                df_sentence.append({
                    'rad_id': item.rad_id,
                    'sentence': sentence,
                })

        df_sentence = pd.DataFrame(df_sentence)
        df_sentence.to_csv(self._sentence_path(), sep='\t', index=False)

    def _word_embedding_path(self):
        return os.path.join(os.getenv('CACHE_DIR'), 'open-i', f'word-embedding-field-{self.field}.pkl')

    def _cider_cache_path(self):
        return os.path.join(os.getenv('CACHE_DIR'), 'open-i', f'cider-cache.pkl')

    def __init__(self,
                 field,
                 min_word_freq,
                 max_report_length,
                 max_sentence_length,
                 embedding_size,
                 phase,
                 kwargs):

        super(OpenIDataset, self).__init__(phase, kwargs)

        self.field = field
        self.min_word_freq = min_word_freq
        self.max_report_length = max_report_length
        self.max_sentence_length = max_sentence_length
        self.embedding_size = embedding_size
        self.phase = phase

        if (phase is None) and (not os.path.isfile(self._meta_path())):
            self._make_meta()

        if OpenIDataset.df_meta is None:
            OpenIDataset.df_meta = pd.read_csv(self._meta_path(), sep='\t')

        if (phase is None) and (not os.path.isfile(self._report_chexpert_path())):
            self._make_report_chexpert()

        if OpenIDataset.df_report is None:
            OpenIDataset.df_report = pd.read_csv(self._report_chexpert_path(), sep='\t')

        if (phase is None) and (not os.path.isfile(self._sentence_path())):
            self._make_sentence()

        if OpenIDataset.df_sentence is None:
            OpenIDataset.df_sentence = pd.read_csv(self._sentence_path(), sep='\t')
            OpenIDataset.label_size = 1

        if (phase is None) and (not os.path.isfile(self._cider_cache_path())):
            self._make_cider_cache()

        if OpenIDataset.df_cache is None:
            OpenIDataset.df_cache = torch.load(self._cider_cache_path())

        if phase == Phase.train:
            if not os.path.isfile(self._word_embedding_path()):
                self._make_word_embedding()

            if OpenIDataset.index_to_word is None:
                word_vectors = KeyedVectors.load(self._word_embedding_path())
                OpenIDataset.index_to_word = np.array(word_vectors.index2entity + [Token.unk])
                OpenIDataset.word_to_index = dict(zip(self.index_to_word, range(len(self.index_to_word))))
                OpenIDataset.word_embedding = np.concatenate([
                    word_vectors.vectors,
                    np.zeros((1, embedding_size)),
                ], axis=0).astype(np.float32)

        df = OpenIDataset.df_meta.merge(OpenIDataset.df_report, on='rad_id', how='inner')
        if phase == Phase.train:
            rad_ids = pd.read_csv(os.path.join(os.getenv('CACHE_DIR'), 'open-i', f'train_ids.csv')).squeeze()
            self.df = df[df.rad_id.isin(rad_ids.values)]
        else:
            rad_ids = pd.read_csv(os.path.join(os.getenv('CACHE_DIR'), 'open-i', f'test_ids.csv')).squeeze()
            self.df = df[df.rad_id.isin(rad_ids.values)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        df_rad = self.df.loc[self.df.rad_id == item.rad_id]

        _item = {
            'item_index': index,
        }

        if self.mode & Mode.enc_image:
            path = os.path.join(os.getenv('CACHE_DIR'), 'open-i', 'images', f'{item.dicom_id}.png')
            image = PIL.Image.open(path)
            image = self.transform(image)

            view_position_index = self.view_position_to_index.get(item.view_position, 0)
            view_position_indices = [self.view_position_to_index.get(view_position, 0) for view_position in df_rad.view_position]

            view_position = torch.cat([
                torch.arange(self.view_position_size) == view_position_index,
                (torch.arange(self.view_position_size).unsqueeze(1) == torch.as_tensor(view_position_indices)).any(1),
            ], 0)
            view_position = torch.as_tensor(view_position, dtype=torch.float)

            _item.update({
                'image': image,
                'view_position': view_position,
            })
        else:
            _item.update({
                'image': torch.randn((self.image_size * self.image_size, self.hidden_size), dtype=torch.float),
                'view_position': torch.randint(0, 2, (2 * self.view_position_size,), dtype=torch.float),
            })

        text = []
        sent_length = []

        df_sentence = self.df_sentence.loc[self.df_sentence.rad_id == item.rad_id]
        if self.mode & Mode.as_one_sentence:
            sentences = [' '.join(df_sentence.sentence)]
        else:
            sentences = df_sentence.sentence

        for sentence in sentences:
            words = sentence.split()

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

        df_chexpert_label = self.df_report.loc[self.df_report.rad_id == item.rad_id].squeeze()
        chexpert_label = torch.as_tensor(df_chexpert_label[CATEGORIES].values.astype(np.int64), dtype=torch.long)

        num = torch.arange(text_length, dtype=torch.long).unsqueeze(1)
        stop = torch.as_tensor(num == text_length - 1, dtype=torch.float)

        _item.update({
            'text_length': text_length,
            'text': text,
            'label': torch.zeros([text_length, self.label_size], dtype=torch.float),
            'chexpert_label': chexpert_label,
            'stop': stop,
            'sent_length': sent_length,
        })

        return _item
