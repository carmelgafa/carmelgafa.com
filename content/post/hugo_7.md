---
title: "Hugo Frontmatter"
date: 2019-12-30T12:01:10+01:00
tags: [hugo_cms]
---

Frontmatter is added as a header when creating a new Hugo file via **hugo new file_name** it has the following format:

```
---
title: "Frontmatter"
date: 2019-12-30T12:01:10+01:00
draft: true
---
```

1. It contains information about the file in key/value pairs.

2. Can be written in JSON TOML, YAML

    * toml is selected. Similar to yaml
    * json can be used but heavy

3. custom variables can be added, eg

    * author
    * language
