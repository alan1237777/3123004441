import sys
import os


def compute_lcs(s1, s2):
    m = len(s1)
    n = len(s2)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
        curr = [0] * (n + 1)
    return prev[n]


def main():
    # 定义原始文件和抄袭文件列表
    orig_file = "orig.txt"
    plag_files = [
        "orig_0.8_add.txt",
        "orig_0.8_del.txt",
        "orig_0.8_dis_1.txt",
        "orig_0.8_dis_10.txt",
        "orig_0.8_dis_15.txt"
    ]

    # 定义输出文件
    output_file = "similarity_results.txt"

    # 检查原始文件是否存在
    if not os.path.isfile(orig_file):
        print(f"错误: 原始文件 '{orig_file}' 不存在。")
        return

    # 读取原始文件内容
    try:
        with open(orig_file, 'r', encoding='utf-8') as f:
            orig_text = f.read()
    except Exception as e:
        print(f"读取原始文件出错: {e}")
        return

    m = len(orig_text)

    # 处理每个抄袭文件
    results = []
    for plag_file in plag_files:
        # 检查抄袭文件是否存在
        if not os.path.isfile(plag_file):
            print(f"警告: 抄袭文件 '{plag_file}' 不存在，已跳过。")
            results.append((plag_file, "文件不存在"))
            continue

        # 读取抄袭文件内容
        try:
            with open(plag_file, 'r', encoding='utf-8') as f:
                plag_text = f.read()
        except Exception as e:
            print(f"读取抄袭文件 {plag_file} 出错: {e}")
            results.append((plag_file, f"读取错误: {e}"))
            continue

        # 计算相似度
        if m == 0:
            similarity = 0.0
        else:
            lcs_len = compute_lcs(orig_text, plag_text)
            similarity = lcs_len / m * 100

        result = round(similarity, 2)
        results.append((plag_file, result))
        print(f"处理完成: {plag_file} -> 相似度: {result:.2f}%")

    # 将结果写入输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("抄袭文件相似度检测结果\n")
            f.write("=" * 40 + "\n")
            f.write(f"原始文件: {orig_file}\n\n")

            for file_name, result in results:
                if isinstance(result, str):
                    f.write(f"{file_name}: {result}\n")
                else:
                    f.write(f"{file_name}: {result:.2f}%\n")

        print(f"\n所有结果已保存到: {output_file}")
    except Exception as e:
        print(f"写入输出文件 {output_file} 出错: {e}")


if __name__ == '__main__':
    main()