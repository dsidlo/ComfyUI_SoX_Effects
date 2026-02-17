# ComfyUI-sox-effects Design

## Implement SoxApplyEffectsNode Multi-Layered Graph Incrementally

### 1. Loop Through Effects Stack and Plot Plottable Effects

- Create a function called `get_plottable_effects` within the `SoxApplyEffectsNode` class.
- Iterate through the effects stack in the `SoxApplyEffectsNode` class.
- For each effect, check if it is plottable (i.e., has a visual representation).
- If the effect is plottable, add it to a list of plottable effects.
- Return the list of plottable effects for further processing.
- Create a test to ensure the function works as expected.
- Do nothing else.

### 2. Implement Plottable Effects Visualization

- Create a function called `get_gnuplot_formulas` within the `SoxApplyEffectsNode` class.
- Use the list of plottable effects obtained from `get_plottable_effects` to generate visual representations.
- Given the list of plottable effects, generate .gnu files of each effect and extract the gnuplot formula, limiting parameters, and step, from each .gnu plot file.
- Return structured collection of the effect, effect parameters, gnuplot formula, limiting parameters, and step for each effect.
- Create a test to ensure the function works as expected.
- Do nothing else.
