(function () {
  document.addEventListener('DOMContentLoaded', function() {
      if (document.body.innerHTML.match(/(\\\(|\\\[|\\begin{|\\frac)/)) {
          // Configure MathJax first
          window.MathJax = {
              startup: {
                  ready: () => {
                      MathJax.startup.defaultReady();
                      MathJax.startup.promise.then(() => {
                          MathJax.typesetPromise();
                      });
                  }
              },
              tex: {
                  inlineMath: [['\\(', '\\)']],
                  displayMath: [['\\[', '\\]']],
                  processEscapes: true,
                  packages: {'[+]': ['ams']}
              },
              options: {
                  ignoreHtmlClass: 'tex-ignore',
                  processHtmlClass: 'tex-process'
              }
          };

          // Load MathJax with proper async handling
          const script = document.createElement('script');
          script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
          script.async = true;
          
          // Add load event handler
          script.onload = () => {
              if (window.MathJax && window.MathJax.typesetPromise) {
                  MathJax.typesetPromise();
              }
          };

          document.head.appendChild(script);
      }
  });
})();