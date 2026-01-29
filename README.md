# ComfyUI_SoX_Effects

This repository contains a collection of ComfyUI nodes for applying various audio effects using the SoX library. These nodes allow users to manipulate audio files within the ComfyUI framework, enhancing the capabilities of generative AI models for audio processing tasks.

![ComfyUI_SoX_Effects Workflow Image](workflows/sox-test-1-no-workflow-color 2026-01-28 01-31-04.png)
<img src="example/sox-test-1-no-workflow-color%202026-01-28%2001-31-04.png?raw">

## Effects supported (61)

## How the nodes work

Load-Audio ---> Sox-Effect-1 --- Sox-Effect-N ---> Sox-Apply ---> Save-Audio

Each socks effect node has an audio and sox-parameters input and an audio-out and sox-parameters output. These are essentially passthroughs where audio is not altered at each effect but passes through instead. On the other hand, sox-parameters for each effect stack up.

The audio is not processed until the **Sox-Apply** node is hit where all the prior stack of effects coming in on sox-parameters are finally applied in the same sequence as the workflow graph, to the audio coming in on its audio input.

Things are a little bit different for the Sox-Spectogram node, in that it will process a copy of the audio up to that point in the workflow and generate a histogram image. But it will pass the unaltered audio through to the next node, and it does not add its own parameters to the current sox-parameters stack and allows it to pass through unaltered.

So the first effects node takes incoming audio from a load-audio node and takes not incoming sox-parameters. The next effect node takes the last effects node's audio and sox-parameters as input and so on until a Sox-Apply node is used to add apply the workflows stack of effects to the audio and finally passes on the processed audion to something like a save-audio node.

## Menu Grouping of Effects

I mainly dove into the SoX rabbit hole to leverage SoX Effects for AI character voice post-processing. So a subset of Effects most likely used to perform voice alteration is listed under "Voice" all other effects are listed under "Effects," and Audio Utility nodes under "Utilities."

## Debugging
