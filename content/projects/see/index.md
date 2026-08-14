+++
title = "see"
+++
{{ project_header(project="see") }}

[`see`](https://github.com/denehoffman/see) is a deliberately small terminal image viewer for terminals with real graphics support. It sends images through Kitty graphics, iTerm2 inline images, or SIXEL rather than approximating them with ASCII, braille, or block characters.

Protocol detection is automatic. The default installation supports PNG and JPEG, with additional image formats available through Cargo feature flags.

## Usage

```console
see image.png
see image.png --full-width
see image.png --original
see image.png --width 800 --height 600
see a.png b.jpg
```

Images can be fitted to the terminal width with vertical scrolling, displayed at their original size, or resized into a width, height, or pixel box while preserving aspect ratio.

The project is intentionally complete in scope: future work is limited to compatibility and support fixes for terminals implementing the supported graphics protocols.
