import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

"""
将文本转换为 BPE（Byte Pair Encoding）编码的 token 序列。
一个高效的 BPE 分词器，支持文本清洗、编码和解码。
"""

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    # 将字节字符映射到Unicode字符，这是一种抽象的字符和代码点的映射关系
    # UTF-8是Unicode的一种实现方式
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    # 返回所有相邻的字符对
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    # 修复Unicode字符问题
    text = ftfy.fix_text(text)
    # 去除html的转义字符
    text = html.unescape(html.unescape(text))
    # 去除收尾空格
    return text.strip()


def whitespace_clean(text):
    # 清理多余空格
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        # 初始化字节编码器和解码器
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 加载BPE编码表
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        # 构建词汇表
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        # 构建编码器和解码器字典
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 缓存特殊token
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        """
        BPE 编码：
            将单词拆分为符号对（bigrams）。
            使用预定义的 BPE 排序（bpe_ranks）来选择最小的符号对进行合并。
            递归合并符号对，直到无法进一步合并。
            缓存已处理的单词以提高效率。
        """
        if token in self.cache:
            return self.cache[token]
        # 将单词转换为元组形式，并在最后一个字符后添加特殊标记 </w>，用于表示单词结束
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        
        pairs = get_pairs(word)
        if not pairs:
            return token+'</w>'

        while True:
            # 选择排序最低的符号对进行合并
            # 直到无法进行合并
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        # 添加到缓存中
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):# 分词，提取单词和特殊标记
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8')) # 转化Unicode
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')) # 拆分成BPE序列并转化为ID
        return bpe_tokens

    def decode(self, tokens):
        # 先转为为BPE，然后逆映射为字节序列，最后替换特殊标记
        # encode的完全逆过程
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
