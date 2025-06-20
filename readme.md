
# 项目名称：人工智能环境搭建与部署

## 一、项目简介
本项目旨在搭建一个基于 Python 的人工智能开发环境，支持运行大语言模型（如通义千问 Qwen-7B 和智谱 ChatGLM-6B）。通过本指南，用户可以快速配置所需的开发环境，并进行模型的测试与分析。

## 二、环境搭建

### （一）选择运行方式
根据您的需求，可以选择以下两种运行方式之一：
1. **conda 环境**：推荐在 conda 环境中运行，便于管理依赖和环境。
2. **root 直接操作**：直接在系统根目录下操作，适合熟悉系统管理的用户。

### （二）conda 环境搭建
1. **安装 Miniconda**
   - 如果系统中没有安装 conda，可以通过以下命令手动下载并安装 Miniconda：
     ```bash
     cd /opt/conda/envs
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
     echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
     source ~/.bashrc
     conda --version
     ```
   - 如果下载链接无法访问，请检查链接的合法性，并尝试在合适的网络环境下重新访问。

2. **创建并激活环境**
   - 创建名为 `qwen_env` 的 conda 环境，并安装 Python 3.10：
     ```bash
     conda create -n qwen_env python=3.10 -y
     source /opt/conda/etc/profile.d/conda.sh
     conda activate qwen_env
     ```

### （三）基础依赖安装
1. **安装基础环境**
   - 安装 PyTorch 和 torchvision（CPU 版本）：
     ```bash
     pip install \
       torch==2.3.0+cpu \
       torchvision==0.18.0+cpu \
       --index-url https://download.pytorch.org/whl/cpu
     ```
   - 如果无法访问 PyTorch 的下载链接，请检查网络连接或尝试其他镜像源。

2. **安装基础依赖**
   - 检查 pip 是否能正常联网：
     ```bash
     pip install -U pip setuptools wheel
     ```
   - 安装其他基础依赖：
     ```bash
     pip install \
       "intel-extension-for-transformers==1.4.2" \
       "neural-compressor==2.5" \
       "transformers==4.33.3" \
       "modelscope==1.9.5" \
       "pydantic==1.10.13" \
       "sentencepiece" \
       "tiktoken" \
       "einops" \
       "transformers_stream_generator" \
       "uvicorn" \
       "fastapi" \
       "yacs" \
       "setuptools_scm"
     ```
   - 安装 `fschat`（需要启用 PEP517 构建）：
     ```bash
     pip install fschat --use-pep517
     ```

## 三、运行测试
1. **编写 Python 文件**
   - 根据实验需求，编写 Python 脚本，运行 AI 模型。更换问题只需更改 `prompt`，更换模型需要更改路径。
   - 示例代码：
     ```python
     from transformers import AutoModelForCausalLM, AutoTokenizer

     model_name = "path/to/model"
     tokenizer = AutoTokenizer.from_pretrained(model_name)
     model = AutoModelForCausalLM.from_pretrained(model_name)

     prompt = "请输入您的问题"
     inputs = tokenizer(prompt, return_tensors="pt")
     outputs = model.generate(**inputs)
     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
     print(response)
     ```

2. **对话测试与横向分析**
   - 运行 Python 脚本，测试不同模型对问题的回答，并进行横向对比分析。

## 四、总结与建议
通过本次实验，我们对通义千问 Qwen-7B 和智谱 ChatGLM-6B 两款大语言模型进行了对比测试。以下是测试结果的总结：
- **简单问题**：两款模型都能给出较为出色的回答，但 ChatGLM-6B 在理解能力上略胜一筹。
- **中等难度问题**：通义千问 Qwen-7B-Chat 能够理解问题意图，但答案存在错误；ChatGLM-6B 则未能精准把握问题关键。
- **高难度问题**：通义千问 Qwen-7B-Chat 能给出较为准确的答案，但回答速度有待提升；ChatGLM-6B 回答较为笼统，未能充分结合语境。

**改进建议**：
- 通义千问 Qwen-7B-Chat 需要进一步提升回答的准确性。
- ChatGLM-6B 需要加强问题理解能力，特别是在复杂语境下的分析能力。

## 五、报告
- 其余详细内容在报告文档中查看。

---

希望这份 `README` 文件能够帮助您快速搭建环境并运行项目。
