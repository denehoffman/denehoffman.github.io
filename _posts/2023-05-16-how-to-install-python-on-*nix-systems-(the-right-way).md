---
layout: post
title: "How to install Python on *NIX systems (the right way)"
date: 2023-05-16 14:27:56
description: Some tips and tricks I've learned that should save you from future pain
tags: python shell linux macos
categories: python
---

### Introduction

For over a decade, Python, specifically Python3, has been instrumental for developers, scientists, scripters, and beginner coders alike. However, one annoying part of python is that there is no built-in version management outside of virtual environments, and these have their drawbacks as well. Imagine the following scenario: You've just purchased a brand new Macbook, and you want to start coding in Python. MacOS is a UNIX-like system, so you can easily open up a new terminal window, type `python3` or `/usr/bin/python3`, and be greeted with a REPL running Python 3.9.6 (at/near time of writing). For many, this is fine, and if you don't need anything beyond this, you can stop reading now. However, what if you wanted a different version of Python? The current latest release is 3.11.3, and future versions promise to not only have new features, but they will be faster, [possibly](https://peps.python.org/pep-0659/) [much](https://github.com/faster-cpython/ideas) [faster](https://thechief.io/c/editorial/how-python-is-becoming-faster/). Or, alternatively, you might work with a large collaboration with members who are hesitant to use newer Python versions. My collaboration uses Python 3.6, and there are significant backwards-incompatible changes that have been made since this version was released (some people even use Python 2.7, against my personal recommendations).

So how do you go about installing these versions on your computer? Well you could just [download](https://www.python.org/downloads/) and build it yourself. This is a perfectly good option if you want to stay with the same version for a very long time and don't care about manually changing your `$PATH` any time you want to switch versions. But there is a better way, and it's called "pyenv".

#### What is pyenv?

[Pyenv](https://github.com/pyenv/pyenv) is a fork from a similar project for managing Ruby versions, and as the name suggests, it is a version/environment manager for Python. It provides a command, `pyenv` with various options for setting the current Python version, installing new versions, and setting defaults for the global environment or even specific folders. I'll tell you how to install it, how to use it, and how to add a couple of plugins that I use frequently.

#### How to install pyenv

There are three main methods to install pyenv:

1. You can use [Homebrew](https://brew.sh/):
```sh
brew update
brew install pyenv
```
If you don't already have Homebrew, I would recommend it, at least for MacOS. There's a simple install script which you can run right in the terminal:
```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Note that you should be cautious to run anything which is pulled directly from the internet. I would recommend checking the link and glancing over the install script, just so you're sure nothing fishy is going on. Homebrew is used by *lots* of developers, so I wouldn't worry about this particular link, but it's good practice to use caution.

2. In a similar fashion, you can install it via an automatic installer script:
```sh
curl https://pyenv.run | bash
```
3. You can just clone the GitHub repository directly:
```sh
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```
If you do it this way, you can also compile the `bash` extension by running
```sh
cd ~/.pyenv && src/configure && make -C src
```
If this fails, the pyenv people say it should still run fine, so don't worry too much.

After you install pyenv, you should make sure your shell environment recognizes it by adding the proper directories to your `$PATH` and evaluating pyenv's "init" function. This is shell-dependent, so I would recommend following the documentation [here](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv).

After installation, restart your terminal to make sure the changes go into effect.

#### How to use pyenv

Now that we have pyenv, we should install a new version of Python. Pyenv conveniently includes your system version by default, so even if you don't want to install another version (see my discussion of virtual environments below), you can still use pyenv. However, if you do want another python version, just use `pyenv install` followed by the desired version. To see a list of available versions, run `pyenv install --list` or use tab completion after `pyenv install`.

This makes it easy to install developer versions (with all the proper C headers), as well as alternatives to [CPython](https://github.com/python/cpython) (the modern standard), like [Stackless Python](https://github.com/stackless-dev/stackless/wiki/), [Pyston](https://www.pyston.org/), [PyPy](https://www.pypy.org/), [Miniforge/Mambaforge](https://github.com/conda-forge/miniforge), [Anaconda](https://www.anaconda.com/download), [Miniconda](https://docs.conda.io/en/latest/miniconda.html), [MicroPython](https://micropython.org/), [Jython](https://www.jython.org/), [IronPython](https://ironpython.net/), [GraalPy](https://github.com/oracle/graalpython), [Cinder](https://github.com/facebookincubator/cinder), and [ActivePython](https://www.activestate.com/products/python/).

To switch between versions in a specific shell, you can use the `pyenv shell` command, followed by the version (again, this should work with tab completion). This will set the version for the current terminal window, so if you want to quickly switch between versions, this is a convenient option. Pyenv installs all the versions in the same place, and dynamically chooses the proper path depending on the environment variable `$PYENV_VERSION`. However, if this variable isn't set, it defaults back to the "local" version or "global" version, in that order.

The "local" version can be set by creating a file called `.python-version` which contains a single line identifying the desired version, although the simpler way to do this is by running the `pyenv local` command followed by the version. To "unset" the local version, just delete that file, or run `pyenv local --unset`. When a local version is set, navigating to the parent directory in a terminal will automatically change the Python version. This is extremely useful if you have multiple projects which run on different versions of Python, or if you like having separate virtual environments for each project (more on this later).

Finally, you should select the "global" version, which will be the default if no shell or local version is found by pyenv. This can be set by the command (you guessed it) `pyenv global` followed by the version. By default, when you install pyenv, this will just be the system version.

#### Plugins

There are numerous [plugins for pyenv](https://github.com/pyenv/pyenv/wiki/Plugins), but I will briefly discuss a couple that I use:

1. [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) adds some simple commands for dealing with virtual environments. If you are unfamiliar with virtual environments in Python, check out the [docs here](https://docs.python.org/3/library/venv.html). Essentially, they are self-contained Python installations with their own set of packages, but importantly they can have the same overall version as other virtual environments. If you use these, you might be used to the following workflow:

    a. Create a new virtual environment in a project path:
```sh
python -m venv /path/to/my/project/my-venv
```

    b. When you want to work on the project, run
```sh
source /path/to/my/project/my-venv/bin/activate
```
(additional scripts for `csh` and `fish` exist here too)

    c. When you're done using the virtual environment, just run `deactivate`.

In my opinion, this seems rather annoying. I used to alias the activate command, since it's annoying to type out the path, and if you put your virtual environments in a project directory, you often have to add something to a `.gitignore` to ignore the virtual environment if your want to use subversioning. This plugin simplifies all of that by adding the following command to pyenv:
```sh
pyenv virtualenv <version> <name>
```
Now you can use all of the above pyenv commands but with your own contained version! Additionally, the new environment is located in the same directory as the other regular python environments installed by pyenv, so there's no more clutter in your project folder (or remembering what path you used). The previous workflow now looks like this:

    a. `pyenv virtualenv 3.11 myproject@3.11` (I like to include the version I use in the virtual environment name)

    b. `cd /path/to/my/project && pyenv local myproject@3.11`

Now, every time you enter the project directory, the virtual environment is automatically activated, and when you leave the directory, it gets deactivated! No more fumbling around with activation scripts!

2. I use [pyright](https://github.com/microsoft/pyright) to lint my code (I should write a blog post about my IDE setup eventually), and by default, it won't work well with pyenv's "local" versioning. However, there's an extremely simple fix, a plugin called [pyenv-pyright](https://github.com/alefpereira/pyenv-pyright). Install it with `git`, and then run `pyenv pyright` in the desired directory, and it creates the proper `pyrightconfig.json` to handle everything.

3. While researching this post, I found [pyenv-default-packages](https://github.com/jawshooah/pyenv-default-packages), which provides a convenient way to specify some default packages to install every time you install a new Python version or virtual environment with pyenv. If you are a scientist like me, you probably always want `numpy`, `scipy`, and `matplotlib` installed in any project, and this is a quick way to just do it and forget about it. 

### Conclusion

If you followed this tutorial, you'll hopefully have a clean way to install and switch between versions and virtual environments of Python. In the future, I'll make a post describing my entire Python setup, which includes [LunarVim](https://www.lunarvim.org/) and a set of plugins and scripts to make code-writing simple. Feel free to reach out to me if you run into problems or have questions, I'm happy to help!

> *"Simplicity is the ultimate sophistication"*
>
> Leonardo da Vinci
