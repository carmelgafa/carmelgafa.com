---
title: "Hugo Archetypes"
date: 2019-12-30T12:01:11+01:00
tags: [hugo_cms]
---

When a file is created by default the following frontmatter is created automatically:

```
---
title: "Archetypes"
date: 2019-12-30T12:01:11+01:00
draft: true
---
```

template for this is in **archetypes/default.md**

```
---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
draft: true
---
```

This is an archetype. The default page used when new content is created. It can be modified to include additional frontmatter.

Creating a **dir_name.md** file in the archetypes folder will create a template for the files created in that folder. It will override **default.md**.
