# Assignment 2 使用及提交说明

实际训练结果：独立测试集 10,000 张图片准确率 64.88%，全部 19 项测试通过。
Docker 容器尚未实际验证；代码已上传到原 GitHub 仓库，尚未提交 Canvas。

本项目接着你的 Assignment 1 写，保留 `/generate` 和 `/embedding`，新增 `/classify`。

1. 安装 Python 3.13。终端进入本文件所在目录。
2. 执行 `python3 -m venv .venv`，然后 `source .venv/bin/activate`。
3. 执行 `pip install -r requirements.txt`。第一次会下载 PyTorch、spaCy 等依赖。
4. 执行 `uvicorn app.main:app --port 8000`。
5. 浏览器打开 http://127.0.0.1:8000/docs 。在 POST /classify 点击 Try it out，上传图片再 Execute。
6. 已附训练后的模型，不需要先重新训练。训练指标在 `models/cifar10_cnn.metrics.json`。

如按评分表检查 Docker，先启动 Docker Desktop：

```sh
docker build -t assignment2-fastapi .
docker run --rm -p 8000:80 assignment2-fastapi
```

GitHub：可以把本目录文件合并进你的 Assignment 1 仓库，保留以前的 theory 文件，
添加本次的 PDF 和模型。提交前执行测试，确认 `.venv`、数据集、凭据没有进入仓库。
在你自己的仓库目录中执行：

```sh
git add app train.py tests requirements.txt requirements-dev.txt Dockerfile .dockerignore .gitignore README.md START_HERE_ZH.md VERIFICATION.md models examples theory/Assignment2_Theory.pdf
git commit -m "Add Assignment 2 CIFAR-10 CNN classifier and theory solutions"
git push
```

Canvas：按课程要求上传 `theory/Assignment2_Theory.pdf`，填写可供老师访问的 GitHub 链接。
GitHub 代码已上传；Canvas 仍需单独提交，请检查课程提交页面是否显示成功。

各层输出维度、运行命令和英文说明见 README.md；实际测试记录见 VERIFICATION.md。
