window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
// Mark inline formulas wider than the text column (narrow screens); stylesheets/extra.css lets them scroll.
function markWideFormulas() {
  document.querySelectorAll(".md-typeset span.arithmatex > mjx-container").forEach(m => {
    m.classList.remove("ps-wide");
    const box = m.closest(".md-typeset");
    if (box && m.getBoundingClientRect().width > box.clientWidth) m.classList.add("ps-wide");
  });
}
let wideTimer;
window.addEventListener("resize", () => { clearTimeout(wideTimer); wideTimer = setTimeout(markWideFormulas, 150); });
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise().then(markWideFormulas);
});
