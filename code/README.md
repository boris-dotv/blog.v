

在 chapters/ 目录里写文章。

打开项目根目录的终端，依次执行：
make clean
make html
make files

启动服务器并预览：
cd build/html
python -m http.server

在浏览器里打开 http://localhost:8000