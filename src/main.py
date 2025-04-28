import matplotlib.pyplot as plt
import seaborn
from tqdm import tqdm

from src.dataset import YACLCDataset
from src.data_augmentor import YACLCDataAugmentor

import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('--average_over', type=int, default=1)
argparser.add_argument('--num_steps', type=int, default=10000)
argparser.add_argument('--annotation_num_per_sentence', type=int, default=1)
argparser.add_argument('--allow_exhaustion', action='store_true')
argparser.add_argument('--evaluation_interval', type=int, default=0)

if __name__ == '__main__':
    data_path = 'data/group_articles_type_correction_user_new.txt'
    dataset = YACLCDataset()
    tqdm.write('Reading data...')
    dataset.read_data(data_path, sent_num_limit=50, use_cache=False)
    tqdm.write('Read complete.')
    # dataset.dump_data()
    tqdm.write('Augmenting data...')
    augmentor = YACLCDataAugmentor(dataset)
    augmentor.augment_dataset(processes=8)
    tqdm.write('Augmentation complete.')
    # augmentor.augment_sent(dataset.id2sent[4921])

    # data_path = 'data/group_articles_type_correction_user_new.txt'
    # dataset = YACLCDataset()
    # statistics = dataset.do_statistics(data_path, sent_num_limit=-1, use_cache=False)
    # percents = [max_num / total_num for (max_num, total_num) in statistics]
    # great = [max_num / total_num for (max_num, total_num) in statistics if max_num / total_num > 0.5]
    # print(len(great) / len(percents))
    # plt.hist(percents, bins=10, density=False, histtype='bar', rwidth=0.8, align='mid')
    # plt.xlabel('gold worker / all worker on sent')
    # plt.show()

    # agreement
    # data_path = 'data/group_articles_type_correction_user_new.txt'
    # dataset = YACLCDataset()
    # tqdm.write('Reading data...')
    # dataset.read_data(data_path, sent_num_limit=-1, use_cache=True)
    # tqdm.write('Read complete.')
    # tqdm.write('Calculating agreement...')
    # sent2agreement = dataset.get_agreement()
    # with open('out/stats/sent2agreement.csv', 'w') as writer:
    #     writer.write(f'Sent ID:::Text:::Agreement\n')
    #     for sent, agreement in sent2agreement.items():
    #         writer.write(f'{sent.id}:::{agreement * 100:.04f}:::{"".join(sent.content)}\n')
    # tqdm.write('Complete.')
    #
    # agreements = []
    # with open('out/stats/sent2agreement.csv') as reader:
    #     reader.readline()
    #     for line in reader:
    #         agreements.append(float(line.strip().split(':::')[1]))
    # seaborn.displot(agreements)
    # plt.show()

    # print(sent2agreement)

