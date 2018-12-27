import nltk
import os
import PIL.Image
import torch
import torch.utils.data

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


class Dataset(torch.utils.data.Dataset):
    view_position_to_index = {
        'AP': 1,
        'PA': 2,
        'LL': 3,
        'LATERAL': 3,
    }

    def __init__(self,
                 df,
                 word_to_index,
                 max_report_length,
                 max_sentence_length,
                 is_train):

        self.df = df
        self.max_report_length = max_report_length
        self.max_sentence_length = max_sentence_length
        self.is_train = is_train

        self.sent_tokenizer = PunktSentenceTokenizer()
        self.word_tokenizer = TweetTokenizer()

        self.num_view_position = max(self.view_position_to_index.values()) + 1

        self.word_to_index = {**word_to_index, **{
            Token.pad: 0,
            Token.unk: len(word_to_index) + 1,
            Token.bos: len(word_to_index) + 2,
            Token.eos: len(word_to_index) + 3,
        }}

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
