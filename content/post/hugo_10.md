---
title: "Hugo Taxonomies"
date: 2019-12-30T13:31:51+01:00
tags: [hugo_cms]
---

1. Taxonomies - how content is grouped together. Two default

    * tags - keywords
    * categories - group content
    * to add modify frontmatter to add taxonomies:

```
---
title: "Taxonomies"
date: 2019-12-30T13:31:51+01:00
author: "Carmel"
draft: true
tags: ["Hugo", "development"]
categories: ["Hugo"]
---
```

2. Tags and categories show in list pages

3. List pages are generated for every tag and category

4. Custom Taxonomies
    * taxonomy must exist in theme
    * in config.toml add taxonomy array

```
    [taxonomies]
        tag = "tags"
        category = "categories"
        mood = "moods"
```
