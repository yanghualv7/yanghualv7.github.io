document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("pre>code").forEach(function (block) {
      hljs.highlightBlock(block);
    });
  });
  