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

## Examples

### 1. Influencing the beginning of the sampling process

Interpolate linearly (the default) from `lion` (step 0) to `bird` (step 7) to `girl` (step 10) and stay at `girl` for the rest of the sampling steps:

```
[lion:bird:girl:, 7, 10]
```

![curve1](https://user-images.githubusercontent.com/32277961/214725976-b72bafc6-0c5d-4491-9c95-b73da41da082.gif)

### 2. Influencing the middle of the sampling process

Interpolate using an embeddings bezier curve from `fireball monster` (step 0) to `dragon monster` (step 30 * 0.4 = step 12), while using `seawater monster` as an intermediate control point to steer the curve away during interpolation and get creative results:

```
[fireball:seawater:dragon: , .1, .4:bezier] monster
```

![curve2](https://user-images.githubusercontent.com/32277961/214941229-2dccad78-f856-42bb-ae6b-16b65b273cda.gif)

## Features
- [Prompt interpolation using a curve function](https://github.com/ljleb/prompt-fusion-extension/wiki/Prompt-syntax)
- [Weight interpolation aware of contextual prompt editing](https://github.com/ljleb/prompt-fusion-extension/wiki/Weight-interpolation)
- Complete backwards compatibility with prompt syntax from https://github.com/AUTOMATIC1111/stable-diffusion-webui

## Installation
1. Visit the **Extensions** tab of Automatic's WebUI.
2. Visit the **Install from URL** subtab.
3. Copy and paste `https://github.com/ljleb/prompt-fusion-extension` in the **URL for extension's git repository** textbox.
4. Press the **Install** button. 


## Usage
- Check the [wiki pages](https://github.com/ljleb/fusion/wiki) for the extension documentation.
