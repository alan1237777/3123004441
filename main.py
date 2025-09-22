import sys
import os
import jieba
import numpy as np
from collections import Counter
import math


def calculate_lcs_similarity(text1, text2):
    """
    使用LCS算法计算两段文本的相似度
    """
    n = len(text1)
    m = len(text2)

    if n == 0 or m == 0:
        return 0.0

    # 创建DP表
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # 填充DP表
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[n][m]
    # 相似度 = LCS长度 / 原文长度
    return lcs_length / n


def calculate_cosine_similarity(text1, text2):
    """
    使用余弦相似度算法计算两段文本的相似度
    """
    # 使用jieba分词
    words1 = list(jieba.cut(text1))
    words2 = list(jieba.cut(text2))

    # 创建词汇表
    vocab = set(words1 + words2)

    # 创建词频向量
    vec1 = Counter(words1)
    vec2 = Counter(words2)

    # 构建向量
    vector1 = [vec1.get(word, 0) for word in vocab]
    vector2 = [vec2.get(word, 0) for word in vocab]

    # 计算点积
    dot_product = sum(a * b for a, b in zip(vector1, vector2))

    # 计算模长
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(a * a for a in vector2))

    # 计算余弦相似度
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def calculate_similarity(original_text, plagiarized_text):
    """
    根据文本长度选择合适的算法计算相似度
    """
    # 设置阈值，短文本使用LCS，长文本使用余弦相似度
    threshold = 1000  # 字符数

    if len(original_text) < threshold and len(plagiarized_text) < threshold:
        try:
            return calculate_lcs_similarity(original_text, plagiarized_text)
        except Exception as e:
            print(f"LCS计算失败，使用余弦相似度替代: {e}")
            return calculate_cosine_similarity(original_text, plagiarized_text)
    else:
        try:
            return calculate_cosine_similarity(original_text, plagiarized_text)
        except Exception as e:
            print(f"余弦相似度计算失败，使用LCS替代: {e}")
            return calculate_lcs_similarity(original_text, plagiarized_text)


def read_file(file_path):
    """
    读取文件内容，处理可能的异常
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise Exception(f"文件不存在: {file_path}")
    except PermissionError:
        raise Exception(f"没有权限读取文件: {file_path}")
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read().strip()
        except:
            raise Exception(f"无法解码文件: {file_path}")
    except Exception as e:
        raise Exception(f"读取文件时发生错误: {e}")


def write_result(output_path, similarity):
    """
    将结果写入文件，处理可能的异常
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("{:.2f}".format(similarity))
    except PermissionError:
        raise Exception(f"没有权限写入文件: {output_path}")
    except Exception as e:
        raise Exception(f"写入文件时发生错误: {e}")


def main():
    """
    主函数，处理命令行参数和主要逻辑
    """
    if len(sys.argv) < 4:
        print("Usage: python paper_check.py <original_file_path> <plagiarized_file_path> <output_file_path>")
        sys.exit(1)

    orig_path = sys.argv[1]
    plag_path = sys.argv[2]
    output_path = sys.argv[3]

    try:
        # 读取文件
        original_text = read_file(orig_path)
        plagiarized_text = read_file(plag_path)

        # 计算相似度
        similarity = calculate_similarity(original_text, plagiarized_text)

        # 写入结果
        write_result(output_path, similarity)

        print(f"相似度计算完成: {similarity:.2f}")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()