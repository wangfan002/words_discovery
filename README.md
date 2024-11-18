# words_discovery 中文新词发现，结合了hanlp
pip install hanlp


## 可以配置多种参数
self.ngram_lengths = [2, 3, 4]  # 要计算的ngram长度
self.mutual_info_threshold = 5  # 互信息阈值
self.entropy_threshold = 1  # 信息熵阈值
self.frequency_threshold = 3  # 词频阈值
self.top_k = {2: 100, 3: 5, 4: 5}  # 针对不同长度的ngram设置前K个新词
self.stopwords = set('第的地得我你他她它年月日时分秒省市县某')  # 停用词
self.symbols = set('（）《》“”‘’【】『』「」。，、；：？！—…')  # 停用符号
self.size = 10000  # 方法字数限制
