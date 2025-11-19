

cd d:/.v.lab/blog.v/code
make clean
make html
make files
cd build/html
python -m http.server

在浏览器里打开 http://localhost:8000

TODO:
代码块颜色需要更换;
增加代码块可复制按钮;