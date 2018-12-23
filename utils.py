import os
import numpy as np
import h5py
import json
import torch
import cv2
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import pandas as pd
import dicom
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class MIMIC_RE(object):
    def __init__(self):
        self._cached = {}

    def get(self, pattern, flags=0):
        key = hash((pattern, flags))
        if key not in self._cached:
            self._cached[key] = re.compile(pattern, flags=flags)

        return self._cached[key]

    def sub(self, pattern, repl, string, flags=0):
        return self.get(pattern, flags=flags).sub(repl, string)

    def rm(self, pattern, string, flags=0):
        return self.sub(pattern, '', string)

    def get_id(self, tag, flags=0):
        return self.get(r'\[\*\*.*{:s}.*?\*\*\]'.format(tag), flags=flags)

    def sub_id(self, tag, repl, string, flags=0):
        return self.get_id(tag).sub(repl, string)

def parse_report(path):
    mimic_re = MIMIC_RE()
    with open(path,'r') as f:
        report = f.read()
    report = report.lower()
    report = mimic_re.sub_id(r'(?:location|address|university|country|state|unit number)', 'LOC', report)
    report = mimic_re.sub_id(r'(?:year|month|day|date)', 'DATE', report)
    report = mimic_re.sub_id(r'(?:hospital)', 'HOSPITAL', report)
    report = mimic_re.sub_id(r'(?:identifier|serial number|medical record number|social security number|md number)', 'ID', report)
    report = mimic_re.sub_id(r'(?:age)', 'AGE', report)
    report = mimic_re.sub_id(r'(?:phone|pager number|contact info|provider number)', 'PHONE', report)
    report = mimic_re.sub_id(r'(?:name|initial|dictator|attending)', 'NAME', report)
    report = mimic_re.sub_id(r'(?:company)', 'COMPANY', report)
    report = mimic_re.sub_id(r'(?:clip number)', 'CLIP_NUM', report)

    report = mimic_re.sub((
        r'\[\*\*(?:'
            r'\d{4}'  # 1970
            r'|\d{0,2}[/-]\d{0,2}'  # 01-01
            r'|\d{0,2}[/-]\d{4}'  # 01-1970
            r'|\d{0,2}[/-]\d{0,2}[/-]\d{4}'  # 01-01-1970
            r'|\d{4}[/-]\d{0,2}[/-]\d{0,2}'  # 1970-01-01
        r')\*\*\]'
    ), 'DATE', report)
    report = mimic_re.sub(r'\[\*\*.*\*\*\]', 'OTHER', report)
    report = mimic_re.sub(r'(?:\d{1,2}:\d{2})', 'TIME', report)

    report = mimic_re.rm(r'_{2,}', report, flags=re.MULTILINE)
    report = mimic_re.rm(r'the study and the report were reviewed by the staff radiologist.', report)


    matches = list(mimic_re.get(r'^(?P<title>[ \w()]+):', flags=re.MULTILINE).finditer(report))
    parsed_report = {}
    for (match, next_match) in zip(matches, matches[1:] + [None]):
        start = match.end()
        end = next_match and next_match.start()

        title = match.group('title')
        title = title.strip()

        paragraph = report[start:end]
        paragraph = mimic_re.sub(r'\s{2,}', ' ', paragraph)
        paragraph = paragraph.strip()
        
        parsed_report[title] = paragraph

    return parsed_report

def iterate_csv(base_path, dataframe, word_freq, max_len, stopword=False):
    image_paths = []
    image_report = []
    image_files = os.listdir(os.path.join(base_path,'images'))
    report_files = os.listdir(os.path.join(base_path,'reports'))
    stop_words = stopwords.words('english')
    my_new_stop_words = ['the','and','to','of','was','with','a','on','in','for','name',
                 'is','patient','s','he','at','as','or','one','she','his','her','am',
                 'were','you','pt','pm','by','be','had','your','this','date',
                'from','there','an','that','p','are','have','has','h','but','o',
                'namepattern','which','every','also','should','if','it','been','who','during', 'x']
    stop_words.extend(my_new_stop_words)
    stop_words = set(stop_words)
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

    for idx, row in dataframe.iterrows():
        if (str(row['dicom_id']) + '.dcm') in image_files and (str(row['rad_id']) + '.txt') in report_files:
            report = []
            report_path = os.path.join(base_path,'reports',str(row['rad_id']) + '.txt')
            path = os.path.join(base_path, 'images', str(row['dicom_id']) + '.dcm')

            parsed_report = parse_report(report_path)
            if 'findings' in parsed_report:
                sentences = tokenizer.tokenize(parsed_report['findings'])
                for sentence in sentences:
                    tokens = word_tokenize(sentence)
                    if stopword:
                        filtered_tokens = [w for w in tokens if not w in stop_words]
                    else:
                        filtered_tokens = tokens
                    word_freq.update(filtered_tokens)
                    if len(filtered_tokens) <= max_len:
                        report.append(filtered_tokens)
                if len(report) == 0:
                    continue

                image_paths.append(path)
                image_report.append(report)

    return image_paths, image_report


def create_input_files(dataset, base_path, min_word_freq, output_folder,
                       max_len=100):

    assert dataset in {'mimiccxr'}

    # Read mimic-cxr-map file
    data = pd.read_csv(os.path.join(base_path, 'mimic-cxr-map.csv'), sep=',', header=0)
    data = data.loc[data['dicom_is_available'],:]
    
    # Split data into three set
    data['random'] = np.random.uniform(0.0,1.0,len(data))
    train = data[data['random'] < 0.7]
    other = data[data['random'] >= 0.7]
    val = other[other['random'] < 0.9]
    test = other[other['random'] >= 0.9]
    train.to_csv('/crimea/liuguanx/mimic-output/train.csv')
    val.to_csv('/crimea/liuguanx/mimic-output/val.csv')
    test.to_csv('/crimea/liuguanx/mimic-output/test.csv')


    # Read image paths and reports for each image
    word_freq = Counter()

    train_image_paths, train_image_sentences = iterate_csv(base_path,train,word_freq,max_len)
    val_image_paths, val_image_sentences = iterate_csv(base_path,val,word_freq,max_len)
    test_image_paths, test_image_sentences = iterate_csv(base_path,test,word_freq,max_len)

    # Sanity check
    assert len(train_image_paths) == len(train_image_sentences)
    assert len(val_image_paths) == len(val_image_sentences)
    assert len(test_image_paths) == len(test_image_sentences)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] >= min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Save images to HDF5 file, and report and their lengths to JSON files
    for impaths, imcaps, split in [(train_image_paths, train_image_sentences, 'TRAIN'),
                                   (val_image_paths, val_image_sentences, 'VAL'),
                                   (test_image_paths, test_image_sentences, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and reports, storing to file...\n" % split)

            total_enc_sentences = []
            total_senlens = []

            for i, path in enumerate(tqdm(impaths)):

                # Assume only one report per image
                sentences = imcaps[i]
                
                # Read images
                plan = dicom.read_file(impaths[i],stop_before_pixels=False)
                img = np.uint8(plan.pixel_array/plan.pixel_array.max()*255)
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                enc_sentences = []
                senlens = []
                for j, c in enumerate(sentences):
                    # Encode sentences
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find report lengths
                    c_len = len(c) + 2

                    enc_sentences.append(enc_c)
                    senlens.append(c_len)
                total_enc_sentences.append(enc_sentences)
                total_senlens.append(senlens)

            # Sanity check
            assert images.shape[0] == len(total_enc_sentences) == len(total_senlens)

            # Save encoded reports and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_REPORT_' + base_filename + '.json'), 'w') as j:
                json.dump(total_enc_sentences, j)

            with open(os.path.join(output_folder, split + '_SENLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(total_senlens, j)

def create_wordmap(min_word_freq, output_folder, stopword=False):
    findings_path = '/data/medg/misc/interpretable-report-gen/data/reports-field-findings.tsv'
    train_path = './data/train.csv'

    train_set = pd.read_csv(train_path)
    rad_ids_list = list(train_set['rad_id'])
    dataframe = pd.read_table(findings_path)
    base_filename = 'mimiccxr_' + str(min_word_freq) + '_min_word_freq'

    stop_words = stopwords.words('english')
    my_new_stop_words = ['the','and','to','of','was','with','a','on','in','for','name',
                 'is','patient','s','he','at','as','or','one','she','his','her','am',
                 'were','you','pt','pm','by','be','had','your','this','date',
                'from','there','an','that','p','are','have','has','h','but','o',
                'namepattern','which','every','also','should','if','it','been','who','during', 'x']
    stop_words.extend(my_new_stop_words)
    stop_words = set(stop_words)
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    word_freq = Counter()

    for idx, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        if row['rad_id'] in rad_ids_list:
            report = row['text']
            sentences = tokenizer.tokenize(report)
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                if stopword:
                    filtered_tokens = [w for w in tokens if not w in stop_words]
                else:
                    filtered_tokens = tokens
                word_freq.update(filtered_tokens)
            
    words = [w for w in word_freq.keys() if word_freq[w] >= min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

if __name__ == "__main__":
    create_wordmap(1,'/crimea/liuguanx/mimic-output')
    # create_input_files('mimiccxr','/crimea/mimic-cxr',1,'/crimea/liuguanx/mimic-output')