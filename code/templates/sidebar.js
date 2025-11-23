document.addEventListener("DOMContentLoaded", function() {
    // 1. 找到侧边栏容器
    const sidebar = document.getElementById('sidebar-content');
    if (!sidebar) return;

    // 2. 找到正文内容区域 (根据你的模板，id是 content)
    const content = document.getElementById('content');
    if (!content) return;

    // 3. 获取所有标题 h2, h3 (h1通常是文章标题，不放入侧边目录)
    const headers = content.querySelectorAll('h2, h3');
    
    if (headers.length === 0) return;

    const ul = document.createElement('ul');
    
    headers.forEach((header, index) => {
        // 如果标题没有ID，给它生成一个
        if (!header.id) {
            header.id = 'header-' + index;
        }

        const li = document.createElement('li');
        const a = document.createElement('a');
        
        a.href = '#' + header.id;
        a.textContent = header.textContent;
        a.className = 'sidebar-link ' + header.tagName.toLowerCase();
        
        // 点击平滑滚动
        a.addEventListener('click', function(e) {
            e.preventDefault();
            header.scrollIntoView({ behavior: 'smooth' });
        });

        li.appendChild(a);
        ul.appendChild(li);
    });

    // 添加标题
    const title = document.createElement('h3');
    title.textContent = "On This Page";
    title.style.marginTop = "0";
    sidebar.appendChild(title);
    sidebar.appendChild(ul);
});