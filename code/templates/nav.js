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
        <li><a href="https://boris-dotv.github.io/">.v's homapege</a></li>
      </ul>
    </div>

    <div class="section">
      <h3>Introductions</h3>
      <ol start="1">
        <li><a href="/c/01-leetcode.html">Leetcode (hot 100)</a></li>
        <li><a href="/c/02-cs336-LLM.html">CS336</a></li>
        
      </ol>
    </div>

    <div class="section">
      <h3>Problem Setup & Context</h3>
      <ol start="4">
        <li><a href="/c/03-nanochat.html">nanochat</a></li>
        <li><a href="/c/04-deepresearch.html">Deep Research</a></li>
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