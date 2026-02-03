# ComfyUI-sox-effects Design

## Design

A dialog box that automatically detects the available effects and provides a user-friendly interface for applying them.

- use `sox -h to list available effects`
- use `sox --help <effect> to get determine the parameters and usage of the effect`
  - This gives us the list of switches, sliders and input parameters to present on the UI.
  - Group the effects by Available. Depreciated '*', Experimental '+' and LibSox-only '#' in separate groups
  - Available effects are shown first.
  - 