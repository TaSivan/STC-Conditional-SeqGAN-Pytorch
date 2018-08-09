# Chinese Short Text Conversation using Conditional SeqGAN

## Prepare dataset
This code use the Sina Weibo dataset release from [STC@NTCIR13](http://ntcirstc.noahlab.com.hk/STC2/stc-cn.htm).

1. `$ python scripts/word_segmentation.py`
    + time: 1:43:05.05
2. `$ python scripts/training_pairs_generation.py`
    + time: 3:21.60

## Prepare pre-trained embedding model

Source:

+ Wiki
    + https://github.com/Kyubyong/wordvectors

+ Weibo
    + https://github.com/Embedding/Chinese-Word-Vectors

1. `$ python scripts/embedding_combine.py`
    + time: 53.610
