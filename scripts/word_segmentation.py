import os
import re
import codecs

from pyhanlp import HanLP, JClass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.join(BASE_DIR, "repository")

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


def ws(filename, convert2zh=False):

    if not os.path.exists(REPO_DIR):
        os.makedirs(REPO_DIR)

    file = os.path.join(REPO_DIR, filename)
    fw = codecs.open(file + '.seg.sc', 'w', encoding = 'utf-8')
    
    regex = re.compile(r'[\u4e00-\u9fffa-zA-Z0-9]+')

    with codecs.open(file, 'r', encoding = 'utf-8') as fr:
        for line in fr:
            line = line.split('\t', 1)[1].strip().replace('“', '').replace('”', '')
            line = clean(line)
            _list = regex.findall(line.strip())
            seq = ''
            for span in _list:
                result = analyzer.analyze(span)
                for terms in result.toSimpleWordList():
                    field = terms.toString().split('/')
                    word = field[0] if not convert2zh else HanLP.convertToTraditionalChinese(field[0])
                    pos = field[1]
                    seq += word.lower() + '_' + pos + ' '

                seq += '，_， '
                    
            fw.write(seq.rsplit('_', 1)[0][:-1] + '。_。\n')

    fw.close()


if __name__ == '__main__':
    # generate `stc2-repos-id-post.seg.sc` and `stc2-repos-id-cmnt.seg.sc`
    ws('stc2-repos-id-post')
    ws('stc2-repos-id-cmnt')


    """

    Before:

        1. 王大姐，打字细心一点
        2. 据说跟女朋友吵架能赢的，最后都单身了。「转」

    After:

        1. 王_nr 大姐_n ，_， 打字_v 细心_a 一点_m 。_。
        2. 据说_v 跟女_n 朋友_n 吵架_d 能_v 赢_v 的_u ，_， 最后_f 都_d 单身_v 了_y 。_。

    Details:
        https://github.com/hankcs/pyhanlp

    """