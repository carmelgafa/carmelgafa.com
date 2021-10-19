---
title: MathJax in Hugo (personal-web theme)
date: "2020-01-08T15:33:41+01:00"
tags: [hugo_cms, mathjax]
---

1. Created a file mathjax_support in **/partials/head/**

```javascript
<script type="text/javascript" async
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
MathJax.Hub.Config({
tex2jax: {
  inlineMath: [['$','$'], ['\\(','\\)']],
  displayMath: [['$$','$$']],
  processEscapes: true,
  processEnvironments: true,
  skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
  TeX: { equationNumbers: { autoNumber: "AMS" },
       extensions: ["AMSmath.js", "AMSsymbols.js"] }
}
});
MathJax.Hub.Queue(function() {
  // Fix <code> tags after MathJax finishes running. This is a
  // hack to overcome a shortcoming of Markdown. Discussion at
  // https://github.com/mojombo/jekyll/issues/199
  var all = MathJax.Hub.getAllJax(), i;
  for(i = 0; i < all.length; i += 1) {
      all[i].SourceElement().parentNode.className += ' has-jax';
  }
});

MathJax.Hub.Config({
// Autonumbering by mathjax
TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>

```

2. In **/partials/head/head.html** added the following line

```javascript
{{ partial "head/mathjax_support.html" . }}
```

3. Example of a post using MathJax

The following code

```latex

\begin{align}
\dot{x} & = \sigma(y-x) \newline
\dot{y} & = \rho x - y - xz \newline
\dot{z} & = -\beta z + xy
\end{align}

\begin{equation}
x=x^2
\end{equation}

$x_2$
```

Generates the following:

\begin{align}
\dot{x} & = \sigma(y-x) \newline
\dot{y} & = \rho x - y - xz \newline
\dot{z} & = -\beta z + xy
\end{align}

\begin{equation}
x=x^2
\end{equation}

$x_2$
