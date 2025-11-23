class NavigationDropdown extends HTMLElement {
  constructor() {
    super();
    const initialExpanded = this.getAttribute('expanded') === 'true';
    
    // ============================================================
    // 🚀 核心修复：智能判断 Base URL
    // ============================================================
    let BASE_URL = '';
    
    // 1. 如果是在 GitHub Pages (域名包含 github.io)
    if (window.location.hostname.includes('github.io')) {
        BASE_URL = '/blog.v'; 
    } 
    // 2. 如果是在本地 (localhost 或 127.0.0.1)，保持为空，不需要前缀
    else {
        BASE_URL = ''; 
    }
    
    // 打印一下，方便你在控制台调试看
    console.log('Current Environment:', window.location.hostname);
    console.log('Determined BASE_URL:', BASE_URL);

    this.innerHTML = `
      <div>
        <button class="dropdown-button" aria-expanded="${initialExpanded}">
          <span><strong>Navigation</strong></span>
          <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M19 9l-7 7-7-7" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
        
        <div class="dropdown-content${initialExpanded ? ' open' : ''}">
          <nav class="chapter-nav">
            <div class="section">
              <h3>Links</h3>
              <ul>
                <!-- 动态拼接路径 -->
                <li><a href="${BASE_URL}/index.html">Home</a></li>
                <li><a href="https://boris-dotv.github.io/">.v's homepage</a></li>
              </ul>
            </div>

            <div class="section">
              <h3>Introductions</h3>
              <ol start="1">
                <li><a href="${BASE_URL}/c/01-leetcode.html">Leetcode (hot 100)</a></li>
                <li><a href="${BASE_URL}/c/02-cs336-LLM.html">CS336</a></li>
              </ol>
            </div>

            <div class="section">
              <h3>Problem Setup & Context</h3>
              <ol start="4">
                <li><a href="${BASE_URL}/c/03-nanochat.html">nanochat</a></li>
                <li><a href="${BASE_URL}/c/04-deepresearch.html">Deep Research</a></li>
              </ol>
            </div>

          </nav>
        </div>
      </div>
    `;

    // 剩下的逻辑保持不变
    const button = this.querySelector('.dropdown-button');
    const content = this.querySelector('.dropdown-content');
    
    button.addEventListener('click', () => {
      const isExpanded = button.getAttribute('aria-expanded') === 'true';
      button.setAttribute('aria-expanded', !isExpanded);
      content.classList.toggle('open');
    });
  }
  
  static get observedAttributes() { return ['expanded']; }
  attributeChangedCallback(name, oldValue, newValue) {
    if (name === 'expanded') {
      const button = this.querySelector('.dropdown-button');
      const content = this.querySelector('.dropdown-content');
      const isExpanded = newValue === 'true';
      if (button && content) {
        button.setAttribute('aria-expanded', isExpanded);
        content.classList.toggle('open', isExpanded);
      }
    }
  }
}

if (!customElements.get('navigation-dropdown')) {
  customElements.define('navigation-dropdown', NavigationDropdown);
}