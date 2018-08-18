import os
import pickle
from tqdm import tqdm
from helper import REPO_DIR

from dataset.gen_dataset import gen_dataset
from dataset.gen_dataloader import get_gen_iter
from models.generator import generator

def write_dis_dataset(gen_iter, generator, filepath):

    if not os.path.exists(REPO_DIR):
        os.makedirs(REPO_DIR)

    iterator = enumerate(gen_iter)

    with open(filepath, "w") as f:
        for _ in tqdm(range(gen_iter.__len__())):
            batch_id, batch_data = next(iterator)
            # - sents:  [ [token] * seq_len ] * batch_size
            # - seqs:   (seq_len, batch_size)
            # - lens:   [1] * batch_size
            query_sents, response_sents, query_seqs, response_seqs, query_lens, response_lens = batch_data
            

            # - out_seqs:           (batch_size, max_out_len)
            # - out_lens:           (batch_size)
            # - decoder_outputs:    (batch_size, max_out_len, vocab_size)
            out_seqs, out_lens, decoder_outputs = generator(query_seqs, query_lens, enable_dropout=False)

            # (seq_len, batch_size) -> (batch_size, seq_len)
            query_seqs = query_seqs.t()         
            response_seqs = response_seqs.t()

            
            # - seq: (max_seq_len)
            real_pairs = list(zip(query_seqs.tolist(), response_seqs.tolist()))  # [(query_seq, response_seq)] * batch_size
            fake_pairs = list(zip(query_seqs.tolist(), out_seqs.tolist()))       # [(query_seq, out_seq)] * batch_size


            out_str = []
            for query, response in real_pairs:
                query_str = []
                for token_id in query:
                    query_str.append(str(token_id))
                response_str = []
                for token_id in response:
                    response_str.append(str(token_id))

                out_str.append("\t".join([" ".join(query_str), " ".join(response_str)]) + "\t1\n")

            for query, response in fake_pairs:
                query_str = []
                for token_id in query:
                    query_str.append(str(token_id))
                response_str = []
                for token_id in response:
                    response_str.append(str(token_id))

                out_str.append("\t".join([" ".join(query_str), " ".join(response_str)]) + "\t0\n")

            f.writelines(out_str)

    print('='*100)
    print('Save file to "{}".'.format(filepath))
    print('='*100 + '\n')


def save_dis_dataset_pkl(filepath, tsv):

    def load_tsv(tsv):
        training_pairs = []

        for line in tqdm(open(tsv).read().strip().split('\n')):
            query_seqs, response_seqs, label = line.split('\t')

            query_seqs = [ int(seq) for seq in query_seqs.split() ]
            response_seqs = [ int(seq) for seq in response_seqs.split() ]
            label = int(label)

            training_pairs.append(((query_seqs, response_seqs), label))

        return training_pairs

    training_pairs = load_tsv(tsv)

    with open(filepath, "wb") as f:
        pickle.dump(training_pairs, f)

    return training_pairs


def load_dis_dataset_pkl(filepath):
    with open(filepath, "rb") as f:
        training_pairs = pickle.load(f)

    return training_pairs


def get_training_pairs():
    # For the sake of reusability, write into a file rather than generate negative example everytime

    TSV_FILE = os.path.join(REPO_DIR, "dis_dataset.tsv")
    PKL_FILE = os.path.join(REPO_DIR, "dis_dataset.pkl")

    if os.path.exists(PKL_FILE):
        training_pairs = load_dis_dataset_pkl(PKL_FILE)
    elif os.path.exists(TSV_FILE):
        training_pairs = save_dis_dataset_pkl(PKL_FILE, TSV_FILE)
    else:    
        gen_iter = get_gen_iter(gen_dataset=gen_dataset, batch_size=64, num_workers=4)
        write_dis_dataset(gen_iter, generator, TSV_FILE)
        training_pairs = save_dis_dataset_pkl(PKL_FILE, TSV_FILE)

    return training_pairs