import unittest
import tempfile
import os
from paper_check import calculate_lcs_similarity, calculate_cosine_similarity, calculate_similarity, read_file, \
    write_result


class TestPaperCheck(unittest.TestCase):

    def test_lcs_similarity_identical(self):
        """测试完全相同文本的LCS相似度"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        similarity = calculate_lcs_similarity(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_lcs_similarity_different(self):
        """测试完全不同文本的LCS相似度"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "明天是星期一，天气阴，我明天要在家休息。"
        similarity = calculate_lcs_similarity(text1, text2)
        self.assertLess(similarity, 0.5)

    def test_lcs_similarity_partial(self):
        """测试部分相似文本的LCS相似度"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "今天是周天，天气晴朗，我晚上要去看电影。"
        similarity = calculate_lcs_similarity(text1, text2)
        # 预期相似度应该在0.5到0.9之间
        self.assertGreater(similarity, 0.5)
        self.assertLess(similarity, 0.9)

    def test_cosine_similarity_identical(self):
        """测试完全相同文本的余弦相似度"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        similarity = calculate_cosine_similarity(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_cosine_similarity_different(self):
        """测试完全不同文本的余弦相似度"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "明天是星期一，天气阴，我明天要在家休息。"
        similarity = calculate_cosine_similarity(text1, text2)
        self.assertLess(similarity, 0.5)

    def test_cosine_similarity_partial(self):
        """测试部分相似文本的余弦相似度"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "今天是周天，天气晴朗，我晚上要去看电影。"
        similarity = calculate_cosine_similarity(text1, text2)
        # 预期相似度应该在0.5到0.9之间
        self.assertGreater(similarity, 0.5)
        self.assertLess(similarity, 0.9)

    def test_calculate_similarity_short_text(self):
        """测试短文本自动选择LCS算法"""
        text1 = "短文本测试"
        text2 = "短文本检查"
        # 两个短文本，应该使用LCS算法
        similarity = calculate_similarity(text1, text2)
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)

    def test_calculate_similarity_long_text(self):
        """测试长文本自动选择余弦相似度算法"""
        # 生成长文本
        text1 = "长文本测试。" * 100  # 约1000字符
        text2 = "长文本检查。" * 100  # 约1000字符
        # 两个长文本，应该使用余弦相似度算法
        similarity = calculate_similarity(text1, text2)
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)

    def test_read_file_existing(self):
        """测试读取存在的文件"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
            f.write("测试文件内容")
            temp_path = f.name

        try:
            content = read_file(temp_path)
            self.assertEqual(content, "测试文件内容")
        finally:
            # 清理临时文件
            os.unlink(temp_path)

    def test_read_file_nonexistent(self):
        """测试读取不存在的文件"""
        with self.assertRaises(Exception):
            read_file("不存在的文件路径.txt")

    def test_write_result(self):
        """测试写入结果到文件"""
        # 创建临时文件路径
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
            temp_path = f.name

        try:
            # 写入结果
            write_result(temp_path, 0.85)

            # 读取并验证结果
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertEqual(content, "0.85")
        finally:
            # 清理临时文件
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()