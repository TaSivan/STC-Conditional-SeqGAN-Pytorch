# Chinese Short Text Conversation using Conditional SeqGAN (Under Construction)

## Dataset
This code use the Sina Weibo dataset release from [STC@NTCIR13](http://ntcirstc.noahlab.com.hk/STC2/stc-cn.htm).

1. `$ python scripts/word_segmentation.py`
    + execution time:  1:43:05.05
2. `$ python scripts/training_pairs_generation.py`
    + execution time:  3:21.60

## Pre-trained embedding model

+ Wiki
    + https://github.com/Kyubyong/wordvectors

+ Weibo
    + https://github.com/Embedding/Chinese-Word-Vectors

1. `$ python scripts/embedding_combine.py`
    + execution time:  53.610