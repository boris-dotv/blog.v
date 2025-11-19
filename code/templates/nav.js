class NavigationDropdown extends HTMLElement {
  constructor() {
    super();
    
    // Get the initial expanded state from the attribute, default to false
    const initialExpanded = this.getAttribute('expanded') === 'true';
    
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
        <li><a href="/">Home</a></li>
        <li><a href="https://github.com/natolambert/rlhf-book">GitHub Repository</a></li>
        <li><a href="/book.pdf">PDF</a> / <a href="https://arxiv.org/abs/2504.12501">Arxiv</a> / <a href="/book.epub">EPUB</a></li>
        <li><a href="https://hubs.la/Q03Tc3cf0">Pre-order now!</a></li>
        <li><a href="/library.html">Example Model Completions</a></li>
      </ul>
    </div>

    <div class="section">
      <h3>Introductions</h3>
      <ol start="1">
        <li><a href="/c/01-introduction.html">Leetcode (hot 100)</a></li>
        <li><a href="/c/02-related-works.html">nanochat</a></li>
        
      </ol>
    </div>

    <div class="section">
      <h3>Problem Setup & Context</h3>
      <ol start="4">
        <li><a href="/c/03-setup.html">Deep Research</a></li>
        <li><a href="/c/04-optimization.html">MTProto</a></li>
      </ol>
    </div>

  </nav>
</div>
</div>
    `;

    // Set up click handler
    const button = this.querySelector('.dropdown-button');
    const content = this.querySelector('.dropdown-content');
    
    button.addEventListener('click', () => {
      const isExpanded = button.getAttribute('aria-expanded') === 'true';
      button.setAttribute('aria-expanded', !isExpanded);
      content.classList.toggle('open');
    });
  }
  
  // Add attribute change observer
  static get observedAttributes() {
    return ['expanded'];
  }
  
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

// Only define the component once
if (!customElements.get('navigation-dropdown')) {
customElements.define('navigation-dropdown', NavigationDropdown);
}