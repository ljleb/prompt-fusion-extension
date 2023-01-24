# Fusion

Fusion is an [auto1111 webui extension](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Developing-extensions) that adds more possibilities to the native prompt syntax. Among other additions, it enables to interpolate between the embeddings of different prompts continuously:

```
# linear prompt interpolation
[night light:magical forest:5,15]

# catmull-rom curve prompt interpolation
[night light:magical forest:norvegian territory:5,15,25:catmull]

# linear weight interpolation
(fire extinguisher:1.0,2.0)

# prompt-editing-aware weight interpolation
[(fire extinguisher:1.0,2.0)::5]
```

## Features
- [Prompt interpolation using a curve function](https://github.com/ljleb/prompt-fusion-extension/wiki/Prompt-syntax)
- [Weight interpolation aware of contextual prompt editing](https://github.com/ljleb/prompt-fusion-extension/wiki/Weight-interpolation)
- Complete backwards compatibility with prompt syntax from https://github.com/AUTOMATIC1111/stable-diffusion-webui

## Installation
1. Visit the **Extensions** tab of Automatic's WebUI.
2. Visit the **Install from URL** subtab.
3. Copy and paste `https://github.com/ljleb/fusion` in the **URL for extension's git repository** textbox.
4. Press the **Install** button. 


## Usage
- Check the [wiki pages](https://github.com/ljleb/fusion/wiki) for the extension documentation.
