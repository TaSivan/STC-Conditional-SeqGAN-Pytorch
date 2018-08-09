import os
import re
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.join(BASE_DIR, "repository")

regex = re.compile(r'[a-zA-Z]+')

POST_FILE = os.path.join(REPO_DIR, "stc2-repos-id-post.seg.sc")
CMNT_FILE = os.path.join(REPO_DIR, "stc2-repos-id-cmnt.seg.sc")

stc2_repos_id_post_seg = open(POST_FILE).read().strip().split('\n')
stc2_repos_id_cmnt_seg = open(CMNT_FILE).read().strip().split('\n')


"""

In[0]:

    stc2_repos_id_cmnt_seg[0]

Out[0]:

    '王_nr 大姐_n ，_， 打字_v 细心_a 一点_m 。_。',

"""

def create_training_list(seg_data):
    output = []
    for sent in seg_data:
        sent_list = []
        for text in sent.split():
            word = text.split('_')[0]
            if regex.findall(word):
                sent_list.extend([w for w in word])
            else:
                sent_list.append(word)
        output.append(sent_list)
    return output


"""

In[1]:

    create_training_list(['额滴勒_m 个_q 去_v ，_， nokia_nx 完_v 胜iphone_n 。_。'])

Out[1]:

    [['额滴勒',
    '个',
    '去',
    '，',
    'n',
    'o',
    'k',
    'i',
    'a',
    '完',
    '胜',
    'i',
    'p',
    'h',
    'o',
    'n',
    'e',
    '。']]

"""

if __name__ == "__main__":
    stc2_repos_id_cmnt_seg = create_training_list(stc2_repos_id_cmnt_seg)
    stc2_repos_id_post_seg = create_training_list(stc2_repos_id_post_seg)
    training_pairs_seg = list(zip(stc2_repos_id_post_seg, stc2_repos_id_cmnt_seg))


    # generate `training_pairs_seg.pkl`

    if not os.path.exists(REPO_DIR):
        os.makedirs(REPO_DIR)
        
    SAVE_FILE = os.path.join(REPO_DIR, "training_pairs_seg.pkl")

    with open(SAVE_FILE, "wb") as f:
        pickle.dump(training_pairs_seg, f)