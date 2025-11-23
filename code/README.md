

cd d:/.v.lab/blog.v/code
make clean
make html
make files
cd build/html
python -m http.server

在浏览器里打开 http://localhost:8000

TODO:
1. DPO 推导, GRPO 推导
2. 提交了什么 PR 到 MindSpeed-MM
3. 解决 OOM 的全思路是什么?
