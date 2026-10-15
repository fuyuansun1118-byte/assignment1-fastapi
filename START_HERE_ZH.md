# Assignment 2 使用及提交说明

已按老师网站的训练配置重新训练：batch size 32、Adam 学习率 0.0005、10 轮、
完整 50,000 张训练图片；图片转换为 [0,1]，不额外归一化。
模型仍严格遵守作业 PDF 的两层 CNN、64×64 输入和 100 单元全连接层。
最终训练集准确率 76.42%，独立测试集准确率 65.34%。21 项测试和 Docker 验证通过。
GitHub 和本作业包已更新；Canvas 尚未提交。

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
