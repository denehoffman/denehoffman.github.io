---
layout: post
title: How I Made This Site
date: 2023-04-26 10:06:07
description: A tutorial for making and automating a personal website using Jekyll
tags: code jekyll script shell arch linux python
categories: scripting
toc: true
---
### Getting Started

This guide is mostly for academics who may have coding and scripting experience but might not want to take a whole course in HTML and CSS to create a professional website. I will walk you through the steps I took to create this site, as well as a few scripts I wrote to automate some of its features. I operate on Linux and I use [LunarVim](https://www.lunarvim.org/) as my default terminal editor, so my comments will mostly be specific to those platforms, but can easily be generalized. For this tutorial, I will assume you have some familiarity with a shell environment such as `bash` or `zsh`, as well as `python` for scripting, so aside from installation differences, everything should be more or less platform independent.

### What is Jekyll

[Jekyll](https://jekyllrb.com/) is a static website generator built on [Ruby](https://www.ruby-lang.org/en/). I can fortunately tell you that by following this guide, you will not need to know any Ruby code, although we will be using quite a lot of Ruby "gems" to make this all work. `jekyll` is one of those gems, particularly one which can be used to make and maintain a simple website. You might wonder how this generator is different from most frameworks, and the answer is that it doesn't build any databases or contain administrative functions like [`django`](https://www.djangoproject.com/), instead opting for hosting raw Markdown files for page content. Markdown is very simple to use, and it already includes functionality for $$ \LaTeX $$-like equation formatting.

### Installing Ruby

Maybe you already have a system version of Ruby on your computer (many Linux distros ship with it, MacOS might also, I haven't checked). If you haven't used it before, it's probably out of date, and updating it can be kind of a pain if you don't know what you're doing. The simple solution is [`rbenv`](https://github.com/rbenv/rbenv). If you've used [`pyenv`](https://github.com/pyenv/pyenv), this will be very familiar (and you should use `pyenv`, it's great!). If you have no idea what I'm talking about, both of these programs are version managers for their respective languages (`ruby` and `python`) which also manage the system version related package managers. To install `rbenv`, you can check the site for some of the platform-specific methods, but I prefer to use `git` repositories when I can. This makes the installation very clean:
```bash
git clone https://github.com/rbenv/rbenv.git ~/.rbenv
```
Next, make sure you initialize the program when you open a new shell. For `bash`, just run
```bash
echo 'eval "$(~/.rbenv/bin/rbenv init - bash)"' >> ~/.bashrc
```
Now restart your shell environment to let these changes go into effect. We'll need one more component, [`ruby-build`](https://github.com/rbenv/ruby-build), which allows us to install new versions through the `rbenv` command. This one can also be installed via `git`:
```bash
git clone https://github.com/rbenv/ruby-build.git "$(rbenv root)"/plugins/ruby-build
```
Next, check the latest available versions with
```bash
rbenv install --list
```
and select the latest (for me, at time of writing, this is `3.2.2`). Then I run
```bash
rbenv install 3.2.2
```
You might replace `3.2.2` with a later version here. `ruby` comes with its own package manager "RubyGems", which is available through the `gem` command. However, we will be using a built in submanager called `bundler`. To make sure everything is set properly, run
```bash
rbenv global 3.2.2
```
to set the current global `ruby` version to `3.2.2`. We can check that this worked by running `which ruby`, for which we expect some path like `~/.rbenv/shims/ruby`.

While the `ruby` installation you just made should have `bundler` pre-installed, we want to make sure it's up to date, which we can do by running
```bash
gem update bundler
```

### Generating a Jekyll Site

We are going to use GitHub Pages to host the website (we'll assume you don't have any extremely large files to upload and don't mind the site being open-source, since the repo needs to be public if you don't pay for premium). If you haven't used GitHub Pages before, it makes this whole process very simple. First, let's create a new directory. For this example, I'll make it `~/my_website/`. In a new file, `~/my_website/Gemfile`, we are going to write the following text:
```
# Gemfile for a Jekyll site with github-pages
source 'https://rubygems.org'
gem 'github-pages', group: :jekyll_plugins
```

Now, while we're inside the `~/my_website/` directory, run
```bash
bundle install
```
If you set up `rbenv` correctly, this should work without any errors, otherwise, make sure you're calling the correct `bundle` command with `which bundle`. This should now give you access to the `jekyll` command. We can create a new site with
```bash
bundle exec jekyll new personal_website
```
This will make a new site in `~/my_website/personal_website`. Next, `cd personal_website` and run
```bash
bundle install
bundle exec jekyll serve
```
If everything is set up correctly, you can now go to [`http://localhost:4000`](http://localhost:4000) to see a barebones website. Any local edits you make will be reflected on this site (except changing the `_config.yml` global settings). However, you may be unlucky like me and run into some error loading `webrick`. This is just a local HTTP server, and you can install it by either running `gem install webrick` or by adding the line `gem 'webrick', '~> 1.8', '>= 1.8.1'` to the Gemfile located at `~/my_website/personal_website/Gemfile`. You should use the latest version, and that information can be found [here](https://rubygems.org/gems/webrick/).

You can check out the configuration file `_config.yml` for some site settings. Every variable here can be accessed in HTML via tags like {% raw %}`{{ site.title }}`{% endraw %}. In fact, if I type that string here without wrapping it with some text to make it render as raw code, it will work right in Markdown (my `site.title` is `{{ site.title }}`). However, I want to include as little work as possible here, so for the next part of this tutorial, we will be using a template. However, this should give you some of the basic ideas of starting a generated website. For a more in-depth look, check out [this blog post by Tania Rascia](https://www.taniarascia.com/make-a-static-website-with-jekyll/), she does a great job with showing how to modify and upload the site to GitHub Pages.

### Using the `al-folio` Template

For my site, I used a template designed for academic websites called [`al-folio`](https://github.com/alshedivat/al-folio). This template is convenient for many reasons, but the most obvious is that it simplifies the build and deploy process using GitHub Actions. To make this all work, we are going to click "Use this template" on the repo's page and select "Create a new repository". Alternately, you can click [here](https://github.com/alshedivat/al-folio/generate).

To properly use GitHub Pages, you need to make a public repository with the name `<username>.github.io`, where `<username>` is your GitHub username. For instance, my username is `denehoffman`, so my repo is located at [https://github.com/denehoffman/denehoffman.github.io](https://github.com/denehoffman/denehoffman.github.io). Then clone the new repo locally with
```bash
git clone git@github.com:<username>/<username>.github.io.git
cd <username>.github.io
```
You can now run
```bash
bundle install
bundle exec jekyll serve --lsi -l
```
These last two options enable LSI for improved related posts and LiveReload, which reloads your preview browser every time you make an edit. In `_config.yml`, you'll need to do a couple of edits (I'll only show lines which need to be changed here):
```yaml
# _config.yml
url: https://<username>.github.io
baseurl: ""
```
replacing `<username>` with your GitHub username, of course. You should still use these settings even if you have a custom domain name. Next, pull up your repo on GitHub and go to the "Settings" tab. From there, check the sidebar and select "Pages" under "Code and automation". From here, you can also set a custom domain, if you have one (I bought my domain name through Google Domains, and the process of setting up GitHub Pages through that was fairly straightforward). I would recommend the "Enforce HTTPS" option, although it isn't required. Under "Build and deployment", we select "Source": "Deploy from a branch", and select the `gh-pages` branch. This is important, as the branch which you will make commits and pull requests to is `master`.

Next, still in settings, but under the "Actions/General" sidebar subitem, we want to change "Workflow permissions" to allow "Read and write permissions". If you want to watch the deployment scripts run, you can move from "Settings" to the "Actions" tab and take a look around. Every time you push a commit to the `master` branch, these deployment actions will be triggered. They will automatically build your site, converting Markdown to HTML along with other intermediate formatting steps. Your site should be available at `<username>.github.io` or at your custom domain name, if the DNS search was successful.

#### Some Recommended Edits

The first thing you'll probably want to change are the default pages, which are linked in the navbar of the website. The first thing I did was get rid of the drop-down submenu example located in `_pages/dropdown.md`. You can either delete this file or just modify it so that it says `nav: false` in the header portion. We can similarly disable any of the other page links by doing this, although a user can still directly access them if they type in the corresponding URL, regardless of whether the navbar link exists (delete the Markdown file to get rid of the page entirely). You can also add custom pages by using the format of the Markdown files here.

Next, if you have a `.bib` file for your own publications, you can really quickly populate the publication page by editing `_bibliography/papers.bib`. If you put enough information, some nice citation info and formatting will be generated by the template. You might have noticed a place on the "about" page for featured publications. You can select which papers show up there by adding `selected={true}` to their `.bib` entry (see the existing file for examples or the [README](https://github.com/alshedivat/al-folio/tree/master#publications)). You'll also want to edit `_pages/publications.md`, particularly the `years` field, which should be a list containing the years in which you have publications.

Next, you should change the image located at `assets/img/prof_pic.jpg` to be your own profile picture. You can if you want to edit the filename itself, it gets added in the `_pages/about.md` file, so you can modify the `profile: image: prof_pic.jpg` field there.

In `_pages/cv.md`, you should change the `cd_pdf` field to point to a PDF of your CV, which should be placed in `assets/pdf/<CV file>.pdf`. Then, in `_data/cv.yml`, you'll want to make some edits to replicate your CV in this Markdown format. The example code there is fairly self-explanatory. If you use GitHub a lot, you might want to edit `_data/repositories.yml`, which is where the `_pages/repositories.md` page sources its information.

Finally, you should check out `_config.yml` again and make some edits. First, set some keywords, maybe modify the footer text, set the site title, your name, and email, and scroll down to the commented section on "Social integration". Here, you can fill out the IDs corresponding to many different social media sites and they will get added automatically to the end of your `about` page. I also set `footer_fixed` to `false` to get a static page footer, and set the `rss_icon` to `false` as well, since I don't care to set up an RSS feed right now. You can also change your blog settings here.

### Scripting

Let's now assume you've had a chance to play around with the various parts of the site. One particular sharp corner is that new blog posts must begin with the `YYYY-MM-DD-` prefix in their filename. I'm still not entirely sure why this is, but we can write a simple script that will automatically fill this stuff out when we want to make a new blog post. We'll make that script in `python` using a package called `rich`. I will assume you already have `python` installed and can run `pip install rich` to install the `rich` package.

What do we want this script to do? I want to make a script that, when run, creates a file in `_posts/` with a well-formatted name that's related to the post title (which we should prompt for) and the date (we can ask `python` for this one). We can also prompt the user for some metadata and even drop them into an editor.

I wrote a short script to populate my post metadata for me. We start with a `rich` console:
```python
#!/usr/bin/env python3

from rich.console import Console
from rich.prompt import PromptBase, Prompt
from pathlib import Path
from subprocess import call
from datetime import datetime
import re
import os

class NamePrompt(PromptBase[str]):
    '''
    A custom prompt which will only validate strings longer than 5 characters
    '''
    @staticmethod
    def check(value: str) -> str:
        output = str(value)
        if len(output) < 5:
            raise ValueError
        return output
    response_type = check
    validate_error_message = "[prompt.invalid]Please enter a title longer than 5 characters"


if __name__ == '__main__':
    console = Console() # create the console object for pretty printing
    # Path to _posts directory
    post_dir = Path.home() / Path('Documents',
                                  'denehoffman.github.io',
                                  '_posts')
    if not post_dir.is_dir():
        exit(f"Could not locate {post_dir}!")

    console.rule("[bold blue]New Blog Post")
    title = NamePrompt.ask("Enter a blog title") # ask for the title
    title_string = '-'.join(title.lower().split()) # "My custom Blog   Post" -> "my-custom-blog-post"

    description = Prompt.ask("Enter a description for this post")

    now = datetime.now() # get current time
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    date_string = now.strftime("%Y-%m-%d")

    post_path = post_dir / f"{date_string}-{title_string}.md"
    if post_path.is_file():
        exit(f"File already exists: {post_path}")

    # let's collect any existing tags in our posts so we don't forget any
    all_tags = []
    for post in post_dir.glob("*.md"): # iterate through all Markdown posts
        post_text = post.read_text() # clever little pathlib shortcut
        match = re.search(r'^tags:\s*(.*)$', post_text, re.MULTILINE)
        if match:
            all_tags.extend(match.group(1).split())
    # this list will probably have duplicates that we don't care about
    all_tags = sorted(list(set(all_tags)))
    console.print(f"Existing blog tags:\n{', '.join(all_tags)}")
    tags = Prompt.ask("Enter some space-separated tags").split()
    
    # we can do the same thing for categories
    all_categories = []
    for post in post_dir.glob("*.md"): # iterate through all Markdown posts
        post_text = post.read_text() # clever little pathlib shortcut
        match = re.search(r'^categories:\s*(.*)$', post_text, re.MULTILINE)
        if match:
            all_categories.extend(match.group(1).split())
    # this list will probably have duplicates that we don't care about
    all_categories = sorted(list(set(all_categories)))
    console.print(f"Existing blog categories:\n{', '.join(all_categories)}")
    categories = Prompt.ask("Enter some space-separated categories").split()

    post_text = f"""---
layout: post
title: {title}
date: {date_time_string}
description: {description}
tags: {' '.join(tags)}
categories: {' '.join(categories)}
---
"""
    post_path.touch()
    post_path.write_text(post_text)

    # You can set the default behavior however you want,
    # I also set $EDITOR to be 'lvim'
    editor = os.environ.get('EDITOR', 'lvim')
    call([editor, str(post_path)])
```

Now I can run this script from anywhere and it will generate and edit new blog posts. As you might imagine, a quick modification can be used to make the "news" posts, but since those don't usually have as much formatting, I use a much simpler script:

```python
#!/usr/bin/env python3

from rich.console import Console
from rich.prompt import PromptBase
from pathlib import Path
from subprocess import call
from datetime import datetime
import os

class NamePrompt(PromptBase[str]):
    '''
    A custom prompt which will only validate strings longer than 5 characters
    '''
    @staticmethod
    def check(value: str) -> str:
        output = str(value)
        if len(output) < 5:
            raise ValueError
        return output
    response_type = check
    validate_error_message = "[prompt.invalid]Please enter a title longer than 5 characters"


if __name__ == '__main__':
    console = Console() # create the console object for pretty printing
    # Path to _posts directory
    post_dir = Path.home() / Path('Documents',
                                  'denehoffman.github.io',
                                  '_news')
    if not post_dir.is_dir():
        exit(f"Could not locate {post_dir}!")

    console.rule("[bold blue]New News Announcement")
    title = NamePrompt.ask("Enter a title") # ask for the title
    title_string = '-'.join(title.lower().split()) # "My custom News   Post" -> "my-custom-news-post"

    now = datetime.now() # get current time
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    date_string = now.strftime("%Y-%m-%d")

    post_path = post_dir / f"{date_string}-{title_string}.md"
    if post_path.is_file():
        exit(f"File already exists: {post_path}")

    post_text = f"""---
layout: post
title: {title}
date: {date_time_string}
inline: true
related_posts: false
---
"""
    post_path.touch()
    post_path.write_text(post_text)

    # You can set the default behavior however you want,
    # I also set $EDITOR to be 'lvim'
    editor = os.environ.get('EDITOR', 'lvim')
    call([editor, str(post_path)])
```

Finally, I wanted to make a quick and easy script to publish my updates to the site via `git` commits. For this, it's easier to just code a shell script:

```bash
#!/bin/bash
cd ~/Documents/denehoffman.github.io
git add --all
git commit --allow-empty-message -e
git push
cd -
```

### Conclusions

If you made it this far without problems, you should now have a very basic professional website, as well as scripts to generate new blog and news posts and publish them with just a few commands. Note that while you are editing your site, it's very useful to run
```bash
bundle exec jekyll serve --lsi -l
```
so that you get a live preview of your site which refreshes when you make changes. This let's you see all your formatting before anything is published. The strength of Jekyll is its simplicity; there are no database structures to learn, you hardly have to touch HTML or CSS unless you want to, and almost everything is written in Markdown, a very simple but powerful markup language. The `al-folio` template can be used as a starting point to generate some useful content quickly. While working on this site, I went down several paths before landing on Jekyll and found the amount of choice overwhelming. In the end, there's no shame in starting with a template. At the time of writing, my website is still in a development stage. I hope to modify the theme, add some custom pages for my different hobbies and projects, and maybe change the formatting on the homepage.

The takeaway is, you have to start somewhere, or it won't get done!

> *"Have no fear of perfection---you'll never reach it."*
>
> Salvador Dali
