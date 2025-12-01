import urllib.parse
import sys

# 尝试导入 pyperclip，这是一个用于剪贴板操作的库
try:
    import pyperclip
except ImportError:
    print("错误: pyperclip 模块未找到。")
    print("请先安装它: pip install pyperclip")
    sys.exit(1)

def generate_tutor_button(python_code: str) -> str:
    """
    接收 Python 代码字符串，返回一个完整的 HTML 按钮，
    其 data-tutor-src 属性包含了 URL 编码后的 Python Tutor 链接。
    """
    if not python_code or not python_code.strip():
        return "错误：输入的代码为空。"

    # 1. 对 Python 代码进行 URL 编码
    # 我们使用 urllib.parse.quote 来确保所有特殊字符都被正确转换
    encoded_code = urllib.parse.quote(python_code)

    # 2. 构建完整的 Python Tutor URL
    # 这是 Python Tutor iframe 链接的模板
    base_url = "https://pythontutor.com/iframe-embed.html#"
    params = {
        "code": encoded_code,
        "cumulative": "false",
        "curInstr": 0,
        "heapPrimitives": "nevernest",
        "origin": "opt-frontend.js",
        "py": 3,
        "rawInputLstJSON": "[]",
        "textReferences": "false"
    }
    
    # 将参数拼接成查询字符串（尽管在#后，格式类似）
    query_string = "&".join(f"{key}={value}" for key, value in params.items())
    
    full_url = base_url + query_string.replace('code=', 'code=', 1) #确保code在第一位

    # 3. 生成最终的 HTML 按钮
    html_button = f'<button class="visualize-btn" data-tutor-src="{full_url}">\n  可视化 ✨\n</button>'
    
    return html_button

if __name__ == "__main__":
    # 从剪贴板读取内容
    print("正在从剪贴板读取 Python 代码...")
    input_code = pyperclip.paste()
    
    # 生成 HTML 按钮
    button_html = generate_tutor_button(input_code)
    
    if "错误" in button_html:
        print(button_html)
    else:
        # 将结果复制回剪贴板
        pyperclip.copy(button_html)
        print("-" * 50)
        print("✅ 成功！HTML 按钮已生成并复制到你的剪贴板。")
        print("现在你可以直接粘贴到你的 Markdown 文件中了。")
        print("-" * 50)
        # 同时也在终端打印出来，方便预览
        # print("\n--- 预览 ---\n")
        # print(button_html)
        # print("\n--------------\n")