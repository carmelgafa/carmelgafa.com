---
title: "Creating and Organizing Content in Hugo"
date: 2019-12-30T12:01:09+01:00
tags: [hugo]
---

1. to create content open cmd in site and type **hugo new file_name**
2. to create content in a directory open cmd in site and type **hugo new dir_name/file_name**
3. to view demo pages as well **execute hugo server -D**
4. Hugo has two types of content

    * single pages
    * list pages. e.g. homepage

5. list pages are created automatically for the first layer in content. Below that you must specify a list page yourself

    * type **hugo new dir1/dir2/_index.md**
    * new file is created - list contents in dir 2
    * content can be added in **_index.md**
    * adding an **_index.md** in a higher level folder overrides default list page
