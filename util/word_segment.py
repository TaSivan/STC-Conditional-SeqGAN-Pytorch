import re
from pyhanlp import HanLP, JClass


PerceptronLexicalAnalyzer = JClass('com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer')
analyzer = PerceptronLexicalAnalyzer()

clean_re = [re.compile(r'via(.*?)$'), re.compile(r'（[^（]+）$'), re.compile(r'「[^「]+」$'), re.compile(r'\([^\(]+\)$')]


def clean(string):

    for regex in clean_re:
        string = regex.sub('', string)

    string = string.replace('（转）', '')\
             .replace('「转」', '')\
             .replace('图转', '')\
             .replace('（转', '')\
             .replace('(转', '')\
             .replace('【全文】', '')\
             .replace('9GAG', '')
    
    return string

def ws(line, convert2zh=False):
    
    regex = re.compile(r'[\u4e00-\u9fffa-zA-Z0-9]+')
    
    line = line.strip().replace('“', '').replace('”', '')
    line = clean(line)
    _list = regex.findall(line.strip())
    seq = ''
    for span in _list:
        result = analyzer.analyze(span)
        for terms in result.toSimpleWordList():
            field = terms.toString().split('/')
            word = field[0] if not convert2zh else HanLP.convertToTraditionalChinese(field[0])
            seq += word.lower() + " "
        seq += '， '

    return (seq.rsplit('，', 1)[0] + '。').split()


""" 

In[0]:

    text = "没有高考，你拼得过官二代吗？"

In[1]:

    ws(text)

Out[1]:

    ['没有', '高考', '，', '你', '拼', '得', '过官', '二代', '吗', '。']

In[2]:

    ws(text, True)

Out[2]:

    ['沒有', '高考', '，', '你', '拼', '得', '過官', '二代', '嗎', '。']

"""