import nltk
import PIL.Image
import torch
import torch.utils.data

from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
)
from torchvision.transforms import (
    Compose,
    ToTensor,
)


class Dataset(torch.utils.data.Dataset):
    view_position_to_index = {
        'AP': 1,
        'PA': 2,
        'LL': 3,
        'LATERAL': 3,
    }

    def __init__(self,
                 df,
                 mimic_cxr,
                 max_report_length,
                 max_sentence_length):

        self.df = df
        self.mimic_cxr = mimic_cxr
        self.max_report_length = max_report_length
        self.max_sentence_length = max_sentence_length

        self.num_view_position = max(self.view_position_to_index.values()) + 1

        # TODO(stmharry): Load
        self.word_to_index = {
            '<pad>': 0,
            '<start>': 1,
            '<end>': 2,
            '<unk>': 3,
        }

        # TODO(stmharry): ColorJitter
        self.transform = Compose([
            ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]

        path = self.mimic_cxr.image_path(dicom_id=item.dicom_id)
        image = PIL.Image.open(path)
        image = self.transform(image)

        view_position_index = self.view_position_to_index.get(item.view_position, 0)
        view_position = torch.arange(self.num_view_position) == view_position_index
        view_position = torch.as_tensor(view_position, dtype=torch.float)

        text = []
        sent_length = []
        # TODO(stmharry): truncate to max_sentence_length
        for sentence in nltk.sent_tokenize(item.text):
            words = nltk.word_tokenize(sentence)

            num_words = min(len(words), self.max_sentence_length)
            words = words[:num_words]

            words = torch.as_tensor((
                [self.word_to_index['<start>']] +
                [self.word_to_index.get(word, self.word_to_index['<unk>']) for word in words] +
                [self.word_to_index['<end>']] +
                [self.word_to_index['<pad>']] * (self.max_sentence_length - num_words)
            ), dtype=torch.long)

            text.append(words)
            sent_length.append(num_words + 2)

        text = torch.stack(text, 0)
        sent_length = torch.as_tensor(sent_length, dtype=torch.long)
        text_length = torch.as_tensor(sent_length.numel(), dtype=torch.long)

        # TODO(stmharry): load label, remember bos/eos!
        label = torch.ones((text_length, 16), dtype=torch.float)

        return {
            'image': image,
            'view_position': view_position,
            'text_length': text_length,
            'text': text,
            'label': label,
            'sent_length': sent_length,
        }
