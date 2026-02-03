@ Sox Multiple File Inputs

SoX allows **multiple files** on the command line in several distinct scenarios beyond the basic single-input + single-output case (i.e., not just one audio input file and one output file).

Here are the main conditions where SoX legitimately accepts/requires/uses **more than two filenames** (or treats extra filenames as additional inputs):

### 1. Multiple Input Files + One Output File (Most Common Case)
SoX combines several input audio files into a single output file.  
The way they are combined is controlled by the `--combine` (or `-m`, `-M`, etc.) global option.

- **Default behavior** (no `--combine` flag): **concatenate** (or **sequence** in some contexts) — appends the files one after another.
  ```bash
  sox file1.wav file2.wav file3.wav output.wav
  ```
  → Plays/concatenates file1, then file2, then file3.

- With explicit `--combine` modes:
  - `--combine concatenate` (or default for `sox` in most usages): appends in sequence.
  - `--combine sequence` : similar, but used more for `play`.
  - `--combine mix` (shorthand `-m`): mixes (adds) the audio signals together, sample-by-sample (useful for layering sounds).
    ```bash
    sox -m drums.wav bass.wav guitar.wav mixed.wav
    ```
  - `--combine mix-power`: similar to mix but scales to avoid clipping more conservatively.
  - `--combine merge` (shorthand `-M`): stacks each input as separate channels (e.g., two mono → one stereo).
    ```bash
    sox -M left.wav right.wav stereo.wav
    ```
  - `--combine multiply`: multiplies signals sample-by-sample (ring modulation, etc.).

You can give as many input files as you want (limited only by OS command-line length and open-file limits).

### 2. Multiple Input Files for Special Effects (Especially `remix`)
The `remix` effect can pull channels from **one** input file, but when you have **multiple input files** combined via `-m`/`-M`/etc., `remix` can reference channels across all of them (after combining).

More commonly, people use multiple files with `remix` indirectly via the combiner.

### 3. Multiple Output Files (Splitting One Input into Many Outputs)
SoX can write to **multiple output files** from a single processing chain by using the special pseudo-effect **`newfile`** inside the effects list.

- This switches SoX into "multiple output mode".
- Every time `newfile` is triggered (or at certain points), it starts a new output file.

Example (records and splits on silence):
```bash
sox input.wav -n newfile : restart silence 1 0.1 1% 1 5.0 1% : newfile : restart
```
Or scripted splits:
```bash
sox long.wav output-%n.wav trim 0 30 : newfile : restart trim 30 30 : newfile ...
```

This is the main case where you see multiple **output** filenames (often with patterns like `%n`, `%1`, etc.).

### 4. Playing Multiple Files (with `play`)
When using the `play` command (which is usually just `sox` symlinked/aliased), you can give multiple files and it will play them in sequence (concatenate by default).
Ctrl+C skips to the next file (different from normal `sox` behavior).

```bash
play track01.ogg track02.ogg track03.ogg
```

### Quick Summary Table

| Scenario                          | Example Command                              | # Input Files | # Output Files | Combiner / Trigger                  |
|-----------------------------------|----------------------------------------------|---------------|----------------|-------------------------------------|
| Concatenate inputs                | `sox a.wav b.wav c.wav out.wav`             | ≥2            | 1              | Default / `--combine concatenate`   |
| Mix / overlay inputs              | `sox -m a.wav b.wav out.wav`                | ≥2            | 1              | `-m` / `--combine mix`              |
| Merge to multi-channel            | `sox -M left.wav right.wav stereo.wav`      | ≥2            | 1              | `-M` / `--combine merge`            |
| Split one input to many outputs   | `sox in.wav out% n.wav newfile ...`         | 1             | ≥2             | `newfile` pseudo-effect             |
| Play sequence                     | `play file1 file2 file3`                    | ≥2            | 0 (to speakers)| Default for `play`                  |

In short:  
- Multiple **inputs** → almost always for combining/mixing/concatenating into one output.  
- Multiple **outputs** → only when using `newfile` in the effects chain (or very old/special usages).

If you're seeing multiple files in a command and aren't sure which case it is, paste the exact command — it's usually one of these patterns.

