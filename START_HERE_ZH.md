# 从这里开始

这是从零搭建的 Assignment 1 项目。英文 README 供老师复现，中文说明供你操作。

## 1. 查看作业

- `theory/Probability_Solutions.pdf`：六道题的英文答案和计算步骤。
- `app/main.py`：API 入口。新增接口为 `POST /embedding`。
- `app/embeddings.py`：课堂指定的 spaCy `en_core_web_lg` 词向量。
- `app/bigram_model.py`：课堂 bigram 文本生成。
- `VERIFICATION.md`：实际测试结果和当前限制。

## 2. 在终端启动

先安装 Python 3.12 和 uv（安装链接见英文 README），进入解压后的 assignment1 文件夹：

```sh
cd /你的实际路径/assignment1
uv sync --locked
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

第一次需要联网下载模型和依赖。看到服务器启动成功后，打开：
http://127.0.0.1:8000/docs

展开 `POST /embedding`，点击 Try it out，输入：

```json
{"word": "apple"}
```

点击 Execute，HTTP 200 的结果中应有 300 个向量数值。
再测试 `POST /generate`：

```json
{"start_word": "the", "length": 20}
```

生成长度可能不足 20，这是遇到没有后继词时的正常停止行为。
终端按 Control+C 可关闭服务器。

## 3. Docker

Docker 已安装，镜像已构建，容器中的健康检查、词向量和文本生成接口均已验证。
以后需要重新构建或启动时，先打开 Docker Desktop，再在项目目录执行：

```sh
docker build -t assignment1-fastapi .
docker run --rm --name assignment1-api -p 8000:80 assignment1-fastapi
```

仍然打开上面的 `/docs` 地址并测试两个接口。使用 8000 端口前，先关闭本地服务器。
两份 PDF 说 Docker 可选，但截图评分标准有 20 分 Docker 项，因此建议验证完成。

## 4. 提交 GitHub 和 Canvas

1. 用你的 GitHub 账号新建一个空仓库。先不要自动添加 README 或 .gitignore。
2. 按英文 README 的 Git 命令提交项目，替换其中的用户名和仓库名。
3. 确保老师有权限访问；不要提供密码或访问令牌给聊天。
4. 在 Canvas 上传概率题 PDF，并在提交说明中附 GitHub 链接；如老师有更具体的
   提交方式，以老师要求为准。
5. 提交前阅读、理解答案和代码；项目注明了 AI 辅助，按课程要求保留或补充披露。

GitHub 仓库：https://github.com/fuyuansun1118-byte/assignment1-fastapi
尚未向 Canvas 提交。代码、依赖锁文件、文档和题解均在此文件夹。
