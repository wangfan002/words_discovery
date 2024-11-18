import collections
import math

import hanlp


# 参数配置
class Config:
    def __init__(self):
        self.ngram_lengths = [2, 3, 4]  # 要计算的ngram长度
        self.mutual_info_threshold = 5  # 互信息阈值
        self.entropy_threshold = 1  # 信息熵阈值
        self.frequency_threshold = 3  # 词频阈值
        self.top_k = {2: 100, 3: 5, 4: 5}  # 针对不同长度的ngram设置前K个新词
        self.stopwords = set('第的地得我你他她它年月日时分秒省市县某')  # 停用词
        self.symbols = set('（）《》“”‘’【】『』「」。，、；：？！—…')  # 停用符号
        self.size = 10000


config = Config()


# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


# 使用hanlp识别人名
def get_person_names(text):
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    # 进行命名实体识别（NER）
    ner_results = HanLP(text, tasks='ner/msra')

    # 提取人名
    persons = {entity[0] for entity in ner_results['ner/msra'] if entity[1] == 'PERSON'}
    return persons


# 生成n-gram词频统计
def generate_ngrams(text, n):
    ngrams = collections.defaultdict(int)
    for i in range(len(text) - n + 1):
        ngram = text[i:i + n]
        ngrams[ngram] += 1
    return ngrams


# 计算互信息
def calculate_mutual_information(ngram_freq, unigram_freq, total_count):
    mutual_info = {}
    for ngram, count in ngram_freq.items():
        if len(ngram) > 1:
            p_ngram = count / total_count
            p_components = 1
            for char in ngram:
                p_components *= (unigram_freq[char] / total_count)
            mutual_info[ngram] = math.log(p_ngram / p_components, 2)
    return mutual_info


# 计算信息熵
def calculate_entropy(ngram_freq):
    total_count = sum(ngram_freq.values())
    entropy = {}
    for ngram, count in ngram_freq.items():
        p_ngram = count / total_count
        entropy[ngram] = -p_ngram * math.log(p_ngram, 2)
    return entropy


# 计算左右熵
def calculate_left_right_entropy(text, ngram_freq, n):
    left_entropy = collections.defaultdict(list)
    right_entropy = collections.defaultdict(list)

    for i in range(len(text) - n):
        ngram = text[i:i + n]
        if i > 0:
            left_entropy[ngram].append(text[i - 1])
        if i + n < len(text):
            right_entropy[ngram].append(text[i + n])

    left_entropy_result = {}
    right_entropy_result = {}
    for ngram in ngram_freq:
        left_entropy_result[ngram] = calculate_single_entropy(left_entropy[ngram])
        right_entropy_result[ngram] = calculate_single_entropy(right_entropy[ngram])

    return left_entropy_result, right_entropy_result


def calculate_single_entropy(neighbors):
    freq = collections.Counter(neighbors)
    total_count = sum(freq.values())
    entropy = 0
    for count in freq.values():
        p = count / total_count
        entropy -= p * math.log(p, 2)
    return entropy


# 检查词语中是否包含符号
def contains_symbol(word):
    return any(char in config.symbols for char in word)


# 检查词语中是否包含停用词
def contains_stopword(word):
    # 定义中文数字列表
    chinese_digits = {'一', '二', '三', '四', '五', '六', '七', '八', '九', '十'}
    # 检查单词中是否包含停用词
    if any(char in config.stopwords for char in word):
        return True
    # 检查单词中是否包含阿拉伯数字
    if any(char.isdigit() for char in word):
        return True
    # 检查单词中是否包含中文数字
    if any(char in chinese_digits for char in word):
        return True
    return False


# 主函数
def progress(text):
    # 去掉空格和回车，保留前5000个字符
    text = content.replace(' ', '').replace('\n', '')[:config.size]
    # 获取人名
    person_names = get_person_names(text)
    # 生成unigram词频统计
    unigram_freq = generate_ngrams(text, 1)
    # 生成不同长度的ngram词频统计
    all_ngram_freq = [generate_ngrams(text, n) for n in config.ngram_lengths]
    # 总字符数
    total_count = len(text)
    # 计算互信息
    mutual_info_results = [calculate_mutual_information(ngram_freq, unigram_freq, total_count) for ngram_freq in
                           all_ngram_freq]
    # 计算左右熵
    entropies_results = [calculate_left_right_entropy(text, ngram_freq, n) for n, ngram_freq in
                         zip(config.ngram_lengths, all_ngram_freq)]

    # 输出结果
    for n, (mutual_info, (left_entropy, right_entropy)) in enumerate(zip(mutual_info_results, entropies_results), start=2):
        if n in config.top_k:
            print(f"\nTop {config.top_k[n]} new words of length {n} (excluding person names, symbols, and stopwords):")
            new_words = [(ngram, mutual_info[ngram], left_entropy[ngram], right_entropy[ngram])
                         for ngram in mutual_info if
                         mutual_info[ngram] > config.mutual_info_threshold and
                         left_entropy[ngram] > config.entropy_threshold and
                         right_entropy[ngram] > config.entropy_threshold and
                         ngram not in person_names and
                         not contains_symbol(ngram) and
                         not contains_stopword(ngram) and
                         all_ngram_freq[n - 2][ngram] > config.frequency_threshold]  # 这里使用 frequency_threshold

            for ngram, mi, le, re in sorted(new_words, key=lambda item: (item[1] + item[2] + item[3]) / 3, reverse=True)[:config.top_k[n]]:
                print(f"{ngram}: MI={mi}, Left Entropy={le}, Right Entropy={re}")


if __name__ == '__main__':
    file_path = 'input.txt'
    content = read_file(file_path)
    progress(content)