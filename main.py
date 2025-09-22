import sys
import os
import jieba
import numpy as np
from collections import Counter
import math
import time
import traceback
import json
from datetime import datetime


class PaperCheckSystem:
    """论文查重系统主类"""

    def __init__(self):
        self.performance_data = {
            "lcs": {"time": 0, "count": 0, "success": 0, "errors": []},
            "cosine": {"time": 0, "count": 0, "success": 0, "errors": []},
            "fallback": {"time": 0, "count": 0, "success": 0, "errors": []}
        }
        self.results_comparison = []

    def calculate_lcs_similarity(self, text1, text2):
        """
        使用LCS算法计算两段文本的相似度
        包含完整的异常处理机制和性能监控
        """
        start_time = time.time()
        try:
            # 处理空文本或单字符文本的特殊情况
            if not text1 or not text2:
                self.performance_data["lcs"]["success"] += 1
                self.performance_data["lcs"]["time"] += time.time() - start_time
                return 0.0

            n = len(text1)
            m = len(text2)

            # 处理单字符文本
            if n == 1 or m == 1:
                if text1[0] == text2[0]:
                    result = 1.0 if n == 1 and m == 1 else 0.5
                else:
                    result = 0.0

                self.performance_data["lcs"]["success"] += 1
                self.performance_data["lcs"]["time"] += time.time() - start_time
                return result

            # 检查文本长度，避免内存溢出
            max_allowed_length = 10000  # 设置最大允许长度
            if n > max_allowed_length or m > max_allowed_length:
                raise MemoryError("文本长度超过LCS算法处理限制")

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

            # 处理除零异常
            if n == 0:
                self.performance_data["lcs"]["success"] += 1
                self.performance_data["lcs"]["time"] += time.time() - start_time
                return 0.0

            # 相似度 = LCS长度 / 原文长度
            similarity = lcs_length / n

            # 确保结果在合理范围内
            result = max(0.0, min(1.0, similarity))

            self.performance_data["lcs"]["success"] += 1
            self.performance_data["lcs"]["time"] += time.time() - start_time
            return result

        except MemoryError as e:
            error_msg = f"LCS内存错误: {e}"
            self.performance_data["lcs"]["errors"].append(error_msg)
            self.performance_data["lcs"]["time"] += time.time() - start_time
            raise
        except Exception as e:
            error_msg = f"LCS计算错误: {e}"
            self.performance_data["lcs"]["errors"].append(error_msg)
            self.performance_data["lcs"]["time"] += time.time() - start_time
            raise ValueError("LCS算法计算失败")

    def calculate_cosine_similarity(self, text1, text2):
        """
        使用余弦相似度算法计算两段文本的相似度
        包含完整的异常处理机制和性能监控
        """
        start_time = time.time()
        try:
            # 处理空文本
            if not text1 or not text2:
                self.performance_data["cosine"]["success"] += 1
                self.performance_data["cosine"]["time"] += time.time() - start_time
                return 0.0

            # 尝试使用jieba分词
            try:
                words1 = list(jieba.cut(text1))
                words2 = list(jieba.cut(text2))
            except Exception as e:
                error_msg = f"分词错误: {e}"
                self.performance_data["cosine"]["errors"].append(error_msg)
                # 分词失败时回退到字符级别
                words1 = list(text1)
                words2 = list(text2)

            # 创建词汇表
            vocab = set(words1 + words2)

            # 处理空词汇表
            if not vocab:
                self.performance_data["cosine"]["success"] += 1
                self.performance_data["cosine"]["time"] += time.time() - start_time
                return 0.0

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

            # 处理除零异常
            if magnitude1 == 0 or magnitude2 == 0:
                self.performance_data["cosine"]["success"] += 1
                self.performance_data["cosine"]["time"] += time.time() - start_time
                return 0.0

            # 计算余弦相似度
            similarity = dot_product / (magnitude1 * magnitude2)

            # 确保结果在合理范围内
            result = max(0.0, min(1.0, similarity))

            self.performance_data["cosine"]["success"] += 1
            self.performance_data["cosine"]["time"] += time.time() - start_time
            return result

        except ZeroDivisionError:
            error_msg = "余弦相似度计算中出现除零错误"
            self.performance_data["cosine"]["errors"].append(error_msg)
            self.performance_data["cosine"]["time"] += time.time() - start_time
            return 0.0
        except Exception as e:
            error_msg = f"余弦相似度计算错误: {e}"
            self.performance_data["cosine"]["errors"].append(error_msg)
            self.performance_data["cosine"]["time"] += time.time() - start_time
            raise ValueError("余弦相似度算法计算失败")

    def fallback_similarity(self, text1, text2):
        """
        降级相似度计算方法
        当所有主要算法都失败时使用
        包含性能监控
        """
        start_time = time.time()
        try:
            if not text1 or not text2:
                self.performance_data["fallback"]["success"] += 1
                self.performance_data["fallback"]["time"] += time.time() - start_time
                return 0.0

            # 简单的字符匹配方法
            common_chars = set(text1) & set(text2)
            total_chars = set(text1) | set(text2)

            if not total_chars:
                self.performance_data["fallback"]["success"] += 1
                self.performance_data["fallback"]["time"] += time.time() - start_time
                return 0.0

            result = len(common_chars) / len(total_chars)

            self.performance_data["fallback"]["success"] += 1
            self.performance_data["fallback"]["time"] += time.time() - start_time
            return result

        except Exception as e:
            error_msg = f"降级算法错误: {e}"
            self.performance_data["fallback"]["errors"].append(error_msg)
            self.performance_data["fallback"]["time"] += time.time() - start_time
            return 0.0

    def calculate_similarity(self, original_text, plagiarized_text, compare_algorithms=False):
        """
        根据文本长度选择合适的算法计算相似度
        包含完整的异常处理机制和性能监控
        """
        # 设置阈值，短文本使用LCS，长文本使用余弦相似度
        threshold = 1000  # 字符数

        # 处理空文本
        if not original_text:
            return 0.0

        # 记录算法比较结果
        comparison_result = {
            "text_length": len(original_text),
            "primary_algorithm": "",
            "primary_result": 0,
            "secondary_algorithm": "",
            "secondary_result": 0,
            "time_difference": 0
        }

        try:
            # 根据文本长度选择算法
            if len(original_text) < threshold and len(plagiarized_text) < threshold:
                comparison_result["primary_algorithm"] = "LCS"

                # 运行主要算法
                primary_start = time.time()
                primary_result = self.calculate_lcs_similarity(original_text, plagiarized_text)
                primary_time = time.time() - primary_start

                comparison_result["primary_result"] = primary_result

                # 如果需要进行算法对比，运行另一种算法
                if compare_algorithms:
                    comparison_result["secondary_algorithm"] = "Cosine"
                    secondary_start = time.time()
                    secondary_result = self.calculate_cosine_similarity(original_text, plagiarized_text)
                    secondary_time = time.time() - secondary_start

                    comparison_result["secondary_result"] = secondary_result
                    comparison_result["time_difference"] = secondary_time - primary_time

                self.results_comparison.append(comparison_result)
                return primary_result
            else:
                comparison_result["primary_algorithm"] = "Cosine"

                # 运行主要算法
                primary_start = time.time()
                primary_result = self.calculate_cosine_similarity(original_text, plagiarized_text)
                primary_time = time.time() - primary_start

                comparison_result["primary_result"] = primary_result

                # 如果需要进行算法对比，运行另一种算法（仅对中等长度文本）
                if compare_algorithms and len(original_text) < 5000:
                    comparison_result["secondary_algorithm"] = "LCS"
                    secondary_start = time.time()
                    secondary_result = self.calculate_lcs_similarity(original_text, plagiarized_text)
                    secondary_time = time.time() - secondary_start

                    comparison_result["secondary_result"] = secondary_result
                    comparison_result["time_difference"] = secondary_time - primary_time

                self.results_comparison.append(comparison_result)
                return primary_result

        except Exception as e:
            print(f"所有算法均失败，使用降级算法: {e}")
            # 终极降级方案
            comparison_result["primary_algorithm"] = "Fallback"
            primary_start = time.time()
            primary_result = self.fallback_similarity(original_text, plagiarized_text)
            primary_time = time.time() - primary_start

            comparison_result["primary_result"] = primary_result
            self.results_comparison.append(comparison_result)
            return primary_result

    def read_file(self, file_path):
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

    def write_result(self, output_path, similarity):
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

    def export_performance_report(self, report_path):
        """
        导出性能报告
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "performance_data": self.performance_data,
                "results_comparison": self.results_comparison,
                "summary": {
                    "total_operations": sum([data["count"] for data in self.performance_data.values()]),
                    "success_rate": {
                        "lcs": self.performance_data["lcs"]["success"] / self.performance_data["lcs"]["count"] * 100 if
                        self.performance_data["lcs"]["count"] > 0 else 0,
                        "cosine": self.performance_data["cosine"]["success"] / self.performance_data["cosine"][
                            "count"] * 100 if self.performance_data["cosine"]["count"] > 0 else 0,
                        "fallback": self.performance_data["fallback"]["success"] / self.performance_data["fallback"][
                            "count"] * 100 if self.performance_data["fallback"]["count"] > 0 else 0
                    },
                    "average_time": {
                        "lcs": self.performance_data["lcs"]["time"] / self.performance_data["lcs"]["count"] if
                        self.performance_data["lcs"]["count"] > 0 else 0,
                        "cosine": self.performance_data["cosine"]["time"] / self.performance_data["cosine"]["count"] if
                        self.performance_data["cosine"]["count"] > 0 else 0,
                        "fallback": self.performance_data["fallback"]["time"] / self.performance_data["fallback"][
                            "count"] if self.performance_data["fallback"]["count"] > 0 else 0
                    }
                }
            }

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"性能报告已导出到: {report_path}")

        except Exception as e:
            print(f"导出性能报告时发生错误: {e}")

    def main(self, orig_path, plag_path, output_path, compare_algorithms=False, report_path=None):
        """
        主函数，处理命令行参数和主要逻辑
        """
        try:
            # 初始化性能计数器
            for algorithm in self.performance_data:
                self.performance_data[algorithm]["count"] = 0

            # 读取文件
            original_text = self.read_file(orig_path)
            plagiarized_text = self.read_file(plag_path)

            # 计算相似度
            similarity = self.calculate_similarity(original_text, plagiarized_text, compare_algorithms)

            # 写入结果
            self.write_result(output_path, similarity)

            print(f"相似度计算完成: {similarity:.2f}")

            # 导出性能报告
            if report_path:
                self.export_performance_report(report_path)

            # 打印性能摘要
            self.print_performance_summary()

        except Exception as e:
            print(f"错误: {e}")
            return 1

        return 0

    def print_performance_summary(self):
        """打印性能摘要"""
        print("\n=== 性能摘要 ===")
        for algo, data in self.performance_data.items():
            if data["count"] > 0:
                success_rate = data["success"] / data["count"] * 100
                avg_time = data["time"] / data["count"] * 1000  # 转换为毫秒
                print(
                    f"{algo.upper()}算法: 调用次数={data['count']}, 成功率={success_rate:.1f}%, 平均耗时={avg_time:.2f}ms")

        if self.results_comparison:
            print("\n=== 算法对比 ===")
            for i, comp in enumerate(self.results_comparison):
                if comp["secondary_algorithm"]:
                    diff = abs(comp["primary_result"] - comp["secondary_result"])
                    time_diff = comp["time_difference"] * 1000  # 转换为毫秒
                    print(f"对比 {i + 1}: {comp['primary_algorithm']}={comp['primary_result']:.3f}, "
                          f"{comp['secondary_algorithm']}={comp['secondary_result']:.3f}, "
                          f"差异={diff:.3f}, 时间差={time_diff:.2f}ms")


def main():
    """
    命令行入口函数
    """
    if len(sys.argv) < 4:
        print(
            "Usage: python paper_check.py <original_file_path> <plagiarized_file_path> <output_file_path> [--compare] [--report <report_path>]")
        print("Options:")
        print("  --compare       比较不同算法的结果")
        print("  --report PATH   导出性能报告到指定路径")
        sys.exit(1)

    orig_path = sys.argv[1]
    plag_path = sys.argv[2]
    output_path = sys.argv[3]

    # 解析可选参数
    compare_algorithms = False
    report_path = None

    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == "--compare":
            compare_algorithms = True
            i += 1
        elif sys.argv[i] == "--report" and i + 1 < len(sys.argv):
            report_path = sys.argv[i + 1]
            i += 2
        else:
            print(f"未知参数: {sys.argv[i]}")
            sys.exit(1)

    # 创建系统实例并运行
    system = PaperCheckSystem()
    exit_code = system.main(orig_path, plag_path, output_path, compare_algorithms, report_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()