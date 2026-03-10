练习做的笔记

1. 创建一个虚拟环境
conda create -n rl_env python=3.10


## 自己完成一个算法
1. 环境状态几维？要输出几个动作？
2. 算法公式需要的输入和输出？
3. 按照Batch给模型喂数据（make_batch()，将零散数据变为PyTorch需要的矩阵形式（torch.tensor()）
4. 将业务逻辑翻译成数学公式。

git config --global user.name "zhelang"
git config --global user.email "z1664890190@gmail.com"

git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

删除远程分支
git push origin --delete revise

1. 更新远程地址：执行 git remote set-url origin https://github.com/ZheLang1223/minimalRL.git 以指向新仓库。
2. 配置身份信息：运行 git config --global user.name "zhelang" 和相关 email 命令，确保提交记录包含你的署名。
3. 解决连接问题：若遇网络报错，使用 git config --global http.sslVerify false 临时跳过证书验证或配置代理端口。
4. 配置忽略规则：在根目录创建 .gitignore 文件并写入不需追踪的路径（如 ../../C/），保持工作区清爽。