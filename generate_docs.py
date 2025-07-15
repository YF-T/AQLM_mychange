#!/usr/bin/env python3
import os
import sys

# --- 配置 ---

# 1. 项目根目录名 (需要分析的文件夹的名称)
PROJECT_ROOT_NAME = '.'  # <--- 您正在分析的项目

# 2. 输出的 Markdown 文件名
OUTPUT_MD_FILE = 'code_analysis_report.md'

# 3. 需要包含在文档中的文件扩展名
ALLOWED_EXTENSIONS = ['.py', '.md', '.h', '.c', '.cpp', '.hh', '.json', '.txt', '.sh', '.yml', '.in', '.cfg']

# 4. 需要额外排除的文件列表 (脚本会自动排除自身和以'.'开头的项)
EXCLUDED_FILES = [
    # 您可以在这里添加其他不希望被分析的文件名
    # 例如: 'LICENSE'
    'code_analysis_report.md',  # 输出文件名
    'generate_docs.py',         # 当前脚本文件
]

# 5. 项目描述 (根据您的具体意图定制)
PROJECT_DESCRIPTION = """
这是AQLM量化方法的官方代码仓库。我希望在其基础上，探索一种更适用于长文本、长代码等长推理场景的量化新方法。

该方法的核心思路是进行选择性量化，即识别并**仅仅针对包含信息量更丰富的高熵（high-entropy）token**进行复杂的量化计算，从而在保证模型性能的同时，优化计算效率。为了实现这一目标，我们需要对校准数据的处理方式进行针对性调整。

**现有的方法处理数据集**时，主要采用两种策略：一种是将多个文本拼接成一个长序列后，再进行随机切片 [cite: 45-53, 60-68]；另一种则是从众多文档中逐个随机采样，直到找到足够长的文档再进行切片 [cite: 20-36, 90-106]。

在我的新探索中，**我不希望拼接数据集**，因为这可能无法精确模拟特定任务（如代码生成）的输入分布。我希望直接使用更具针对性的数据，例如来自 **nvidia/OpenMathReasoning** 数据集。具体来说，我计划**直接使用模型生成的解答（`generated_solution`）的一部分**作为校准数据。这种方式不仅能保证数据内容与目标任务的高度相关性，也更符合我对长代码或长逻辑链进行推理量化的最终目标。
"""

# --- 函数定义 ---

def generate_directory_tree(root_path):
    """
    生成指定路径的目录结构树，忽略所有以 '.' 开头的目录和文件。
    """
    tree_lines = []
    
    # 使用 os.walk 来遍历，topdown=True 是默认行为
    for dirpath, dirnames, filenames in os.walk(root_path):
        # --- 核心修改：原地修改 dirnames 列表以阻止 os.walk 进入这些目录 ---
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        
        # 同样过滤掉以 '.' 开头的文件和在排除列表中的文件
        filenames = [f for f in filenames if not f.startswith('.') and f not in EXCLUDED_FILES]
        
        level = dirpath.replace(root_path, '').count(os.sep)
        indent = '│   ' * level
        
        # 打印当前目录的相对路径
        if dirpath != root_path:
            tree_lines.append(f"{indent[:-4]}├── {os.path.basename(dirpath)}/")
            indent += '│   '
            
        for i, filename in enumerate(sorted(filenames)):
            connector = "└── " if i == len(filenames) - 1 else "├── "
            tree_lines.append(f"{indent}{connector}{filename}")
            
    return f"{os.path.basename(root_path)}/\n" + "\n".join(tree_lines)


def collect_file_contents(root_path, allowed_extensions):
    """
    递归收集指定目录下所有符合条件的文件路径和内容，
    忽略所有以 '.' 开头的目录和文件。
    """
    file_data = []
    if not os.path.isdir(root_path):
        return []

    for dirpath, dirnames, filenames in os.walk(root_path):
        # --- 核心修改：原地修改 dirnames 列表以阻止 os.walk 进入这些目录 ---
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]

        filenames.sort()
        for filename in filenames:
            # 排除特定文件和以 '.' 开头的文件
            if filename in EXCLUDED_FILES or filename.startswith('.'):
                continue

            if os.path.splitext(filename)[1].lower() in allowed_extensions:
                file_path_full = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(file_path_full, os.path.dirname(root_path))
                try:
                    with open(file_path_full, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_data.append((relative_path, content))
                except UnicodeDecodeError:
                    print(f"Warning: Could not read file {file_path_full} with UTF-8. Skipping.")
                except Exception as e:
                    print(f"Warning: Error reading file {file_path_full}: {e}")
    return file_data


def write_markdown(output_file, project_name, description_template, tree_string, file_contents):
    """
    将所有收集到的信息写入一个结构化的 Markdown 文件。
    """
    lang_map = {
        '.py': 'python', '.md': 'markdown', '.h': 'c', '.c': 'c', '.cpp': 'cpp',
        '.hh': 'cpp', '.json': 'json', '.sh': 'bash', '.txt': 'text', '.yml': 'yaml',
        '.in': 'text', '.cfg': 'ini'
    }
    
    description = description_template.strip()

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {project_name} 项目代码分析与开发规划\n\n")
            f.write("## 1. 项目概述与开发目标\n\n")
            f.write(description)
            f.write("\n\n")

            f.write("## 2. 原始项目目录结构\n\n")
            f.write("```\n")
            f.write(tree_string)
            f.write("\n```\n\n")
            
            f.write("## 3. 核心源代码分析\n\n")
            if not file_contents:
                f.write("在指定目录中没有找到符合条件的文件。\n")
                return
            
            for relative_path, content in file_contents:
                ext = os.path.splitext(relative_path)[1].lower()
                lang = lang_map.get(ext, '')

                path_for_display = relative_path.replace('\\', '/')
                f.write(f"### `{path_for_display}`\n\n")
                f.write(f"```{lang}\n")
                f.write(content.strip())
                f.write("\n```\n\n")
                
        print(f"成功！文档已生成: {output_file}")
    except IOError as e:
        print(f"错误: 无法写入文件 {output_file}: {e}")

# --- 主程序 ---

if __name__ == "__main__":
    # 动态获取脚本自身的文件名并加入排除列表
    script_filename = os.path.basename(__file__)
    if script_filename not in EXCLUDED_FILES:
        EXCLUDED_FILES.append(script_filename)
    if OUTPUT_MD_FILE not in EXCLUDED_FILES:
        EXCLUDED_FILES.append(OUTPUT_MD_FILE)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.join(script_dir, PROJECT_ROOT_NAME)

    if not os.path.isdir(project_path):
        print(f"错误: 根目录 '{PROJECT_ROOT_NAME}' 在当前路径下未找到。")
        print(f"请确保此脚本与 '{PROJECT_ROOT_NAME}' 文件夹在同一目录中。")
        sys.exit(1)

    print(f"将忽略所有以 '.' 开头的目录和文件，以及在排除列表中的文件: {EXCLUDED_FILES}")

    print("开始生成目录树...")
    tree_str = generate_directory_tree(project_path)
    
    print(f"正在从 '{PROJECT_ROOT_NAME}' 目录收集文件内容...")
    files = collect_file_contents(project_path, ALLOWED_EXTENSIONS)
    
    print(f"正在将分析结果写入 Markdown 文件 '{OUTPUT_MD_FILE}'...")
    write_markdown(OUTPUT_MD_FILE, PROJECT_ROOT_NAME, PROJECT_DESCRIPTION, tree_str, files)