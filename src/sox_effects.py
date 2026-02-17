import shlex
import subprocess
import tempfile
import os
import torch
import torchaudio
import numpy as np
import re
from typing import Optional
import shutil
from typing import Optional

from PIL import Image
# Check if the program is being run by bytest
if os.environ.get('PYTEST_VERSION'):
    from sox_node_utils import SoxNodeUtils
else:
    from .sox_node_utils import SoxNodeUtils


class SoxApplyEffectsNode:
    # Tested: DGS v0.1.3
    """
    TODO: Add the ability to graph all graphable effect and combine them into one graph.
          - calculate and plot the combined Final Net Response.
          - calculate and plot original an audio's characteristics vs and how it is affected by the chain of effects.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_apply": ("BOOLEAN", {"default": True, "tooltip": "Enable application of SoX effects"}),
                "enable_sox_plot": ("BOOLEAN", {
                    "default": False,
                    "tooltip": """Enable SoX --plot mode for diagnostic visualization of transfer functions (gnuplot script → PNG image).

**Important Notes**:
- Only the **first plottable effect** in the SOX_PARAMS chain is visualized (ignores subsequent effects).
- Supported effects (transfer-function based): equalizer, highpass, lowpass, bandpass, bandreject, allpass, sinc, fir, biquad, compand (compression curve), bass, treble.
- If no supported effects in chain → blank/empty plot (minimal script output).
- Useful combinations: Tune single filters (e.g., chain only 'highpass 1000' → plot response curve); preview FIR coeffs.
- Issues to avoid: Multi-effect chains (only first plotted); 
  - non-linear effects (reverb/echo ignored, blank plot); 
  - production audio workflows (breaks processing—use separately for tuning)."""
                }),
                "plot_size_x": ("INT", {
                    "default": 800,
                    "min": 100,
                    "max": 2000,
                    "step": 10,
                    "tooltip": """Plot PNG width in pixels (X-axis).

Default: 800px (good detail/speed balance).
• Smaller (100-400): Quick thumbs, low-res previews.
• Larger (1200-2000): High detail for printing/analysis.

Larger sizes increase render time/memory."""
                }),
                "plot_size_y": ("INT", {
                    "default": 400,
                    "min": 100,
                    "max": 800,
                    "step": 10,
                    "tooltip": """Plot PNG height in pixels (Y-axis, frequency).

Default: 400px (detailed magnitude response).
• Shorter (100-200): Minimal space.
• Taller (400-800): Room for phase plots, legends.

Pairs with X for total canvas size; keep aspect ~3:1."""
                }),
                "save_sox_plot": ("BOOLEAN", {
                    "default": False,
                    "tooltip": """Save the generated sox_plot PNG to disk (incremental numbering to avoid overwrites).

Only active if enable_sox_plot is True. Files saved as `{plot_file_prefix}_{nnnn}.png` (e.g., sox_plot_images/sox_plot_0001.png).
Creates directory if needed; logs full path in dbg-text on success."""
                }),
                "plot_file_prefix": ("STRING", {
                    "default": "output/images/sox_plot_images/sox_plot",
                    "tooltip": """File prefix for saved PNG plots (e.g., 'my_plots/filter').

Defaults to 'sox_plot_images/sox_plot'. Saves incrementally: `{prefix}_{nnnn}.png` (nnnn=0001+).
Only saves if save_sox_plot=True and enable_sox_plot=True. Useful: Organize plots by workflow (e.g., 'fir_tests/coeff1')."""
                }),
                "params": ("SOX_PARAMS",),
            },
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "STRING")
    RETURN_NAMES = ("audio", "sox_plot", "dbg-text")
    FUNCTION = "apply_effects"
    CATEGORY = "audio/SoX/Effects/Apply"
    DESCRIPTION = """Applies the chained SoX effects parameters to the input audio. 
  - If enable_sox_plot=True, generates diagnostic plot PNG (no audio processing; passes through). 
    - *** If enable_sox_plot=True, no audio processing is performed. ***"""

    def apply_effects(self, audio, params, enable_apply=True, enable_sox_plot=False,
                      plot_size_x=800, plot_size_y=400,
                      save_sox_plot=False, plot_file_prefix="sox_plot_images/sox_plot"):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        sox_cmd_params = params.get("sox_params", [])
        cmd_str = "sox input.wav output.wav " + shlex.join(
            sox_cmd_params) if sox_cmd_params else "No effects applied (audio passed through)."

        # Handle plotting if enabled (no audio processing; diagnostic only)
        sox_plot_image = torch.zeros((1, 400, 800, 3), dtype=torch.uint8)  # Blank default
        plot_dbg = ""
        plot_script_path = None
        png_path = None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
            torchaudio.save(temp_input.name, waveform[0], sample_rate)  # Use first batch item
            input_path = temp_input.name
        output_path = tempfile.mktemp(suffix='.wav')  # Dummy output
        # Audio processing (skipped if plotting, per tooltip)
        output_waveforms = []
        processed_audio = audio  # Default passthrough

        plot_dbg = ""
        sox_dbg = ""
        if enable_sox_plot:
            if not sox_cmd_params:
                plot_dbg += "** Plot skipped: Empty SOX_PARAMS chain (no effects to plot). **\n"
            else:
                # Run SoX --plot gnuplot input.wav output.wav [effects] to generate script (with temp files)
                plot_cmd = ['sox', '--plot', 'gnuplot', input_path, output_path] + sox_cmd_params
                plot_script_path = tempfile.mktemp(suffix='.soxplot')
                try:
                    plot_dbg += f"** SoX Plot cmd executed: {shlex.join(plot_cmd)}\n"
                    result = subprocess.run(plot_cmd, capture_output=True, check=False, text=True)
                    if (enable_sox_plot is True and result.returncode != 2) or (enable_sox_plot is False and result.returncode != 0):
                        plot_dbg += f"** SoX Plot cmd failed (rc={result.returncode}); skipping render. **\n"
                    plot_dbg += f"Plot script captured from audio cmd stdout ({len(result.stdout)} chars).\n"
                    if result.stdout.strip():
                        plot_dbg += f"\n--- Sox --plot  STDOUT ---\n{result.stdout.strip()}\n--- Sox --plot STDOUT END ---\n"
                    if result.stderr.strip():
                        plot_dbg += f"\n--- Sox --plot STDERR ---\n{result.stderr.strip()}\n--- Sox --plot STDERR END ---\n"
                    # Filter out 'pause ' lines before saving
                    lines = [line for line in result.stdout.splitlines() if not line.strip().startswith('pause ')]
                    filtered_content = '\n'.join(lines) + '\n'
                    plot_dbg += f"*** Final gnup ****\n"
                    with open(plot_script_path, 'w') as f:
                        f.write(filtered_content)
                    plot_dbg += f"Plot script generated: {plot_script_path}\n--- script start---\n{filtered_content}\n--- script end---\n\n"
                    # Render to PNG
                    png_path = tempfile.mktemp(suffix='.png')
                    plot_dbg += "Calling: SoxNodeUtils.render_sox_plot_to_image()...\n"
                    render_msg, gnuplot_stdout, gnuplot_stderr = SoxNodeUtils.render_sox_plot_to_image(sox_plot_script_path=plot_script_path,
                                                                                                       output_image=png_path,
                                                                                                       x=plot_size_x, y=plot_size_y)
                    if gnuplot_stdout.strip():
                        plot_dbg += f"\n--- GNUPLOT STDOUT ---\n{gnuplot_stdout}\n--- GNUPLOT STDOUT END ---\n"
                    if gnuplot_stderr.strip():
                        plot_dbg += f"\n--- GNUPLOT STDERR ---\n{gnuplot_stderr}\n--- GNUPLOT STDERR END ---\n"

                    # Save if requested (incremental, like spectrogram)
                    if save_sox_plot:
                        plot_dbg += "---> Saving plot...\n"
                        base_prefix = plot_file_prefix.strip()
                        if not base_prefix:
                            plot_dbg += "- Save skipped: Empty plot_file_prefix.\n"
                        else:
                            # The full dir path
                            dir_path = os.path.dirname(os.path.abspath(f"{base_prefix}")) or '.'
                            # Get filename prefix from plot_file_prefix
                            filename_prefix = os.path.basename(base_prefix)
                            plot_dbg += f"- base_prefix: {base_prefix}  dir_path: {dir_path}  filename_prefix: {filename_prefix}\n"
                            os.makedirs(dir_path, exist_ok=True)
                            pattern = rf'^{re.escape(filename_prefix)}_(\d+).png$'
                            nums = []
                            # Collects existing sequence numbers from matching plot files
                            try:
                                for f in os.listdir(dir_path):
                                    plot_dbg += f"   - {f}\n"
                                    m = re.match(pattern, f)
                                    if m:
                                        nums.append(int(m.group(1)))
                            except OSError:
                                pass
                            next_seq = max(nums, default=0) + 1
                            filename = f"{filename_prefix}_{next_seq:04d}.png"
                            full_save_path = os.path.join(dir_path, filename)
                            shutil.copy2(png_path, full_save_path)
                            plot_dbg += f"- filename: {filename}  full_save_path: {full_save_path}\n"
                            plot_dbg += f"--> ...Saved plot: {os.path.abspath(full_save_path)} (seq {next_seq:04d})\n"
                    else:
                        plot_dbg += "Save skipped: save_sox_plot=False.\n"

                    if render_msg is None:
                        # Load PNG as IMAGE tensor
                        plot_dbg += f"==> Opening Rendered Image ({png_path})..."
                        pil_img = Image.open(png_path).convert("RGB")
                        img_array = np.array(pil_img)
                        sox_plot_image = torch.from_numpy(img_array).unsqueeze(0).to(torch.uint8)  # [1, H, W, 3]
                        plot_dbg += f"gnuplot IMAGE ready ({plot_size_x}x{plot_size_y} PNG).\n"
                        plot_dbg += "==> ...Rendered Image is good."
                    else:
                        plot_dbg += f"** gnuplot failed: STDERR follows... **\n--- gnuplot stderr start ---\n{gnuplot_stderr}\n--- gnuplot stderr end ---\n"
                except Exception as e:
                    plot_dbg += f"*** Exception ***: gnuplot Render failed\n{str(e)}\n--- open_plot ---\n{plot_dbg}\n--- open_plot end ---\n\n"
        for i in range(waveform.shape[0]):
            single_waveform = waveform[i]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
                torchaudio.save(temp_input.name, single_waveform, sample_rate)
                input_path = temp_input.name

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
                output_path = temp_output.name

            output_waveforms.append(single_waveform)

            if enable_apply and sox_cmd_params:
                sox_dbg += f"\n*** SoxApplyEffectsNode Enabled ***\n"
                cmd = ['sox', input_path, output_path] + sox_cmd_params
                try:
                    sp_ret = subprocess.run(cmd, capture_output=True, check=False, text=True)
                    if sp_ret.returncode != 0:
                        sox_dbg += f"** SoX cmd executed: {shlex.join(cmd)}\n"
                        sox_dbg += f"** SoX cmd failed (rc={sp_ret.returncode}); skipping render. **\n"
                    out_waveform, _ = torchaudio.load(output_path)
                    output_waveforms[-1] = out_waveform
                    if sp_ret.stdout.strip():
                        sox_dbg += f"\n--- SoX STDOUT ---\n{sp_ret.stdout}\n--- SoX STDOUT END ---\n"
                    if sp_ret.stderr.strip():
                        sox_dbg += f"\n--- SoX STDERR ---\n{sp_ret.stdout}\n--- SoX STDERR END ---\n"
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"\n*** SoX Exception ***: {e.stderr}\n--- sox_debug ---\n{sox_dbg}\n--- soxdbgend ---\n\n")
                sox_dbg += f"\n - sox effects successfully applied to audio.\n"
                sox_dbg += f"\n - sox cmd executed: {shlex.join(cmd)}\n"
            else:
                sox_dbg += f"\n*** SoxApplyEffectsNode NOT Enabled ***: Audio Effects not applied.\n"

            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)

        sox_dbg += "\n--- Prep Output Waveforms... ---\n"
        max_samples = max(w.shape[-1] for w in output_waveforms)
        padded_waveforms = []
        for w in output_waveforms:
            padding = max_samples - w.shape[-1]
            if padding > 0:
                w = torch.nn.functional.pad(w, (0, padding))
            padded_waveforms.append(w)

        stacked = torch.stack(padded_waveforms)

        processed_audio = {"waveform": stacked, "sample_rate": sample_rate}
        sox_dbg += "--- ...Prep Output Waveforms Compelted ---\n"

        # Prepare image_out
        batch_imgs = []
        sox_dbg += "--- Prep Output Image ---\n"
        if enable_apply and enable_sox_plot:
            # PNG exists (/tmp/png_path), load IMAGE first
            try:
                pil_img = Image.open(png_path)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                img_np = np.array(pil_img)
                img_t = (torch.from_numpy(img_np).to(torch.float32) / 255.0).unsqueeze(0)
                batch_imgs.append(img_t)
                sox_dbg += "Successful Output Image preparation.\n"
            except Exception as e:
                sox_dbg += f"\n*** Exception Preparing Output Image:***\n{e}\n\n"
                batch_imgs.append(torch.zeros((1, plot_size_y, plot_size_x, 3), dtype=torch.uint8))
            finally:
                # Cleanup temps
                for path in [input_path, output_path, plot_script_path, png_path]:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
            sox_plot_image = torch.cat(batch_imgs, dim=0)
        sox_dbg += "--- Prep Output Image ---\n"

        # Combine debug
        full_dbg = f"Model Command: {cmd_str}\n\n=== plot_dbg ===\n{plot_dbg}\n=== plot_dbg end ===\n\n=== sox_dbg ===\n{sox_dbg}\n=== sox_dbg end ===\n".strip()

        return (processed_audio, sox_plot_image, full_dbg)

    @staticmethod
    def get_plottable_effects(sox_params):
        """
        Iterate through the effects stack and identify plottable effects.
        
        Plottable effects (transfer-function based) per SoX --plot documentation:
        equalizer, highpass, lowpass, bandpass, bandreject, allpass, 
        sinc, fir, biquad, compand, bass, treble
        
        Args:
            sox_params: List of SoX effect parameters (effect name followed by args)
            
        Returns:
            List of dicts: [{'effect': name, 'args': [arg1, arg2, ...]}, ...]
        """
        # Effects that support --plot visualization
        plottable_names = {
            'equalizer', 'highpass', 'lowpass', 'bandpass', 'bandreject',
            'allpass', 'sinc', 'fir', 'biquad', 'compand', 'bass', 'treble'
        }
        
        # Known effect names to distinguish from arguments
        all_effect_names = {
            'allpass', 'band', 'bandpass', 'bandreject', 'bass', 'bend', 'biquad',
            'channels', 'chorus', 'compand', 'contrast', 'dcshift', 'deemph', 'delay',
            'dither', 'downsample', 'earwax', 'echo', 'echos', 'equalizer', 'fade',
            'fir', 'flanger', 'gain', 'highpass', 'hilbert', 'ladspa', 'loudness',
            'lowpass', 'mcompand', 'noiseprof', 'noisered', 'norm', 'oops', 'overdrive',
            'pad', 'phaser', 'pitch', 'rate', 'remix', 'repeat', 'reverb', 'reverse',
            'riaa', 'silence', 'sinc', 'speed', 'splice', 'stat', 'stats', 'stretch',
            'swap', 'synth', 'tempo', 'treble', 'tremolo', 'trim', 'upsample', 'vad',
            'vol'
        }
        
        plottable_effects = []
        i = 0
        n = len(sox_params)
        
        while i < n:
            token = sox_params[i]
            
            if token in all_effect_names:
                effect_name = token
                i += 1
                
                if effect_name in plottable_names:
                    args = []
                    
                    # Extract arguments based on effect type
                    if effect_name == 'equalizer':
                        # frequency width[q|o|h|k] gain (3 args)
                        for _ in range(3):
                            if i < n and sox_params[i] not in all_effect_names:
                                args.append(sox_params[i])
                                i += 1
                                
                    elif effect_name in ('highpass', 'lowpass'):
                        # [-1|-2] frequency [width] (1-3 args)
                        if i < n and sox_params[i] in ('-1', '-2'):
                            args.append(sox_params[i])
                            i += 1
                        for _ in range(2):  # freq and optional width
                            if i < n and sox_params[i] not in all_effect_names:
                                args.append(sox_params[i])
                                i += 1
                                
                    elif effect_name in ('bandpass', 'bandreject', 'allpass'):
                        # 1-2 args each
                        for _ in range(2):
                            if i < n and sox_params[i] not in all_effect_names:
                                args.append(sox_params[i])
                                i += 1
                                
                    elif effect_name == 'sinc':
                        # [-h] [-n|-t] [-k freq] freq ... (up to 5 tokens)
                        for _ in range(5):
                            if i < n and sox_params[i] not in all_effect_names:
                                args.append(sox_params[i])
                                i += 1
                                
                    elif effect_name == 'fir':
                        # coeffs_file (1 arg)
                        if i < n and sox_params[i] not in all_effect_names:
                            args.append(sox_params[i])
                            i += 1
                            
                    elif effect_name == 'biquad':
                        # frequency gain BW|Q|S [norm] (3-4 args)
                        for _ in range(4):
                            if i < n and sox_params[i] not in all_effect_names:
                                args.append(sox_params[i])
                                i += 1
                                
                    elif effect_name == 'compand':
                        # Complex arguments, collect until next effect
                        while i < n and sox_params[i] not in all_effect_names:
                            args.append(sox_params[i])
                            i += 1
                            
                    elif effect_name in ('bass', 'treble'):
                        # gain [frequency] [width] (1-3 args)
                        for _ in range(3):
                            if i < n and sox_params[i] not in all_effect_names:
                                args.append(sox_params[i])
                                i += 1
                    
                    plottable_effects.append({
                        'effect': effect_name,
                        'args': args
                    })
                else:
                    # Skip arguments for non-plottable effects
                    while i < n and sox_params[i] not in all_effect_names:
                        i += 1
            else:
                i += 1
        
        return plottable_effects

    @staticmethod
    def get_gnuplot_formulas(plottable_effects, sample_rate=44100, wave_file: Optional[str] = None):
        """
        Generate gnuplot formulas for each plottable effect by running SoX --plot.
        
        For each plottable effect, generates a .gnu plot file using SoX --plot,
        then extracts the gnuplot formula, limiting parameters (xrange, yrange),
        and step information.
        
        Args:
            plottable_effects: List of dicts from get_plottable_effects()
            sample_rate: Sample rate for the dummy audio file (default 44100)
            
        Returns:
            List of dicts containing:
            - 'effect': effect name
            - 'args': effect arguments
            - 'gnuplot_formula': the plot formula string
            - 'xrange': x-axis limits [min, max] or None
            - 'yrange': y-axis limits [min, max] or None  
            - 'step': step size or None (extracted from xrange/samples)
        """
        results = []

        input_path = None
        synthetic_created = False
        if wave_file and os.path.exists(wave_file):
            input_path = wave_file
        else:
            try:
                # Create dummy audio file for SoX --plot
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
                    # Create silent 1-second audio
                    dummy_audio = torch.zeros(1, 1, sample_rate, dtype=torch.float32)
                    torchaudio.save(temp_input.name, dummy_audio[0], sample_rate)
                    input_path = temp_input.name
                synthetic_created = True
            except Exception as e:
                raise RuntimeError(f"Failed to create synthetic input for plotting: {str(e)}")

        output_path = tempfile.mktemp(suffix='.wav')
        cmd_base = ['sox', '--plot', 'gnuplot', input_path, output_path]
        for effect_info in plottable_effects:
                effect_name = effect_info['effect']
                args = effect_info['args']
                
                # Build SoX command with just this effect
                plot_cmd = cmd_base + [effect_name] + args
                
                try:
                    result = subprocess.run(plot_cmd, capture_output=True, check=False, text=True)
                    gnuplot_script = result.stdout
                    
                    # Parse the gnuplot script
                    formula_data = SoxApplyEffectsNode._parse_gnuplot_script(gnuplot_script)
                    
                    results.append({
                        'effect': effect_name,
                        'args': args,
                        'gnuplot_formula': formula_data.get('formula', ''),
                        'xrange': formula_data.get('xrange'),
                        'yrange': formula_data.get('yrange'),
                        'step': formula_data.get('step')
                    })
                except Exception as e:
                    # If SoX fails for this effect, still include it with error info
                    results.append({
                        'effect': effect_name,
                        'args': args,
                        'gnuplot_formula': '',
                        'xrange': None,
                        'yrange': None,
                        'step': None,
                        'error': str(e)
                    })
        if synthetic_created:
            try:
                os.remove(input_path)
            except OSError:
                pass
        try:
            os.remove(output_path)
        except OSError:
            pass
        return results

    @staticmethod
    def _parse_gnuplot_script(script):
        """
        Parse a gnuplot script to extract formula, ranges, and step.
        
        Args:
            script: String containing gnuplot commands
            
        Returns:
            Dict with 'formula', 'xrange', 'yrange', 'step'
        """
        formula_data = {
            'formula': '',
            'xrange': None,
            'yrange': None,
            'step': None
        }
        
        lines = script.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Extract xrange
            if line.startswith('set xrange'):
                # Parse "set xrange [min:max]" or "set xrange [*:*]"
                match = re.search(r'\[([^:]+):([^]]+)\]', line)
                if match:
                    xmin_str, xmax_str = match.group(1), match.group(2)
                    try:
                        xmin = float(xmin_str) if xmin_str != '*' else None
                        xmax = float(xmax_str) if xmax_str != '*' else None
                        formula_data['xrange'] = [xmin, xmax]
                        
                        # Calculate step if we have valid range
                        if xmin is not None and xmax is not None:
                            # Default samples is usually 1000 in gnuplot
                            samples = 1000
                            # Check if samples is set explicitly
                            for l in lines:
                                if l.strip().startswith('set samples'):
                                    samples_match = re.search(r'samples\s+(\d+)', l)
                                    if samples_match:
                                        samples = int(samples_match.group(1))
                                        break
                            formula_data['step'] = (xmax - xmin) / samples
                    except ValueError:
                        pass
            
            # Extract yrange
            elif line.startswith('set yrange'):
                match = re.search(r'\[([^:]+):([^]]+)\]', line)
                if match:
                    ymin_str, ymax_str = match.group(1), match.group(2)
                    try:
                        ymin = float(ymin_str) if ymin_str != '*' else None
                        ymax = float(ymax_str) if ymax_str != '*' else None
                        formula_data['yrange'] = [ymin, ymax]
                    except ValueError:
                        pass
            
            # Extract samples setting
            elif line.startswith('set samples'):
                samples_match = re.search(r'samples\s+(\d+)', line)
                if samples_match and formula_data['xrange']:
                    samples = int(samples_match.group(1))
                    xmin, xmax = formula_data['xrange']
                    if xmin is not None and xmax is not None:
                        formula_data['step'] = (xmax - xmin) / samples
            
            # Extract formula from plot command
            # Look for lines like: plot ... title '...' with lines, ...
            elif line.startswith('plot') or line.startswith('replot'):
                # Extract the formula part after 'plot' and before 'title' or 'with'
                # Example: plot [0:22050] 20*log10(abs(1/sqrt((1+($1/1000)**2)))) title 'highpass' with lines
                plot_match = re.search(r'plot\s+(?:\[.*?\]\s+)?(.*?)(?:\s+title|\s+with|$)', line)
                if plot_match:
                    formula_data['formula'] = plot_match.group(1).strip()
                else:
                    # Fallback: just take everything after 'plot'
                    formula_data['formula'] = line[4:].strip()
        
        return formula_data


class SoxAllpassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": """Input audio waveform tensor.
    
    Passes through unchanged; used for chaining to other nodes.
    
    Supports mono/stereo/multi-channel; batched [B,C,T] format."""}),
                "enable_allpass": ("BOOLEAN", {
                    "default": True,
                    "tooltip": """Enable the Allpass filter effect.
    
    Applies phase shift without altering amplitude (all-pass filter).
    
    Useful for: Creating comb filtering, subtle spatial effects, or pre-delay in reverb chains.
    
    Combine with: DelayNode for metallic/comb reverb; avoid high widths (>5) with low frequencies (<500Hz) to prevent instability or artifacts.
    
    SoX syntax: allpass <frequency> <width>[h|k|q|o] (q=quality recommended)."""
                }),
                "allpass_frequency": ("FLOAT", {
                    "default": 1000.0,
                    "min": 0.0,
                    "max": 20000.0,
                    "step": 1.0,
                    "tooltip": """Center frequency (Hz) for the allpass filter.
    
    Determines where phase shift is most pronounced (0-20kHz audible range).
    
    • Low (<500Hz): Bass phase effects (e.g., warmth/thump).
    • Mid (1-5kHz): Vocal/instrumental coloration.
    • High (>5kHz): Airy shimmer (use narrow width to avoid harshness).
    
    Tip: Start at 1000Hz; pair with width=1q for subtle; test with/without DelayNode to avoid muddiness."""
                }),
                "allpass_width": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": """Bandwidth/quality of the allpass filter (Q-factor).
    
    Controls sharpness of phase shift (0.1=wide/broad, 10=narrow/precise).
    
    • Low Q (0.1-1.0): Gentle, smooth phase (good for overall tone shaping).
    • High Q (5-10): Sharp notch-like (risk of ringing; use sparingly).
    
    Units: Appends 'q' (quality); e.g., 1.0 → "1q". Avoid extremes (>8) at high frequencies to prevent digital artifacts or instability in chains."""
                }),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS", {"tooltip": """Previous SoX effects chain (list of params).
    
    Appends allpass params if enabled; passes through for further chaining.
    
    Use: Wire from prior effect nodes (e.g., GainNode → AllpassNode → ApplyEffectsNode)."""}),
            }
        }
    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "SOX_PARAMS", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = """Allpass SoX effect node for chaining sox effects.
   allpass frequency[k] width[h|k|o|q]
          Apply  a  two-pole  all-pass filter with central frequency (in Hz) frequency, 
          and filter-width width.  An all-pass filter changes the audio's frequency to phase 
          relationship without changing its frequency to amplitude
          relationship..
    """

    def process(self, audio, enable_allpass=True, allpass_frequency=1000.0, allpass_width=1.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["allpass", str(allpass_frequency), f"{allpass_width}q"]
        debug_str = shlex.join(effect_params)
        if enable_allpass:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxBandNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_band": ("BOOLEAN", {"default": True, "tooltip": "band [-n] center [width[h|k|q|o]]"}),
                "band_narrow": ("BOOLEAN", {"default": False}),
                "band_center": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "band_width": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 10000.0, "step": 1.0}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Band SoX effect node for chaining. dbg-text STRING: 'band params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_band=True, band_narrow=False, band_center=1000.0, band_width=100.0,
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["band"]
        if band_narrow:
            effect_params.append("-n")
        effect_params += [str(band_center), f"{band_width}Hz"]
        debug_str = shlex.join(effect_params)
        if enable_band:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxBandpassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_bandpass": ("BOOLEAN",
                                    {"default": True, "tooltip": "bandpass [-c center] [width[h|k|q|o]] freq"}),
                "bandpass_frequency": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "bandpass_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Bandpass SoX effect node for chaining. dbg-text STRING: 'bandpass params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_bandpass=True, bandpass_frequency=1000.0, bandpass_width=1.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["bandpass", str(bandpass_frequency), f"{bandpass_width}q"]
        debug_str = shlex.join(effect_params)
        if enable_bandpass:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxBandrejectNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_bandreject": ("BOOLEAN",
                                      {"default": True, "tooltip": "bandreject [-c center] [width[h|k|q|o]] freq"}),
                "bandreject_frequency": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "bandreject_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Bandreject SoX effect node for chaining. dbg-text STRING: 'bandreject params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_bandreject=True, bandreject_frequency=1000.0, bandreject_width=1.0,
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["bandreject", str(bandreject_frequency), f"{bandreject_width}q"]
        debug_str = shlex.join(effect_params)
        if enable_bandreject:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxBiquadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_biquad": ("BOOLEAN", {"default": True, "tooltip": "biquad frequency gain BW|Q|S [norm]"}),
                "biquad_frequency": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "biquad_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "biquad_q": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "biquad_norm": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Biquad SoX effect node for chaining. dbg-text STRING: 'biquad params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_biquad=True, biquad_frequency=1000.0, biquad_gain=0.0, biquad_q=1.0, biquad_norm=1,
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["biquad", str(biquad_frequency), str(biquad_gain), str(biquad_q), str(biquad_norm)]
        debug_str = shlex.join(effect_params)
        if enable_biquad:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxChannelsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_channels": ("BOOLEAN", {"default": True, "tooltip": "channels [number]"}),
                "channels_number": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Channel"
    DESCRIPTION = "Channels SoX effect node for chaining. dbg-text STRING: 'channels params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_channels=True, channels_number=2, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["channels", str(channels_number)]
        debug_str = shlex.join(effect_params)
        if enable_channels:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxContrastNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_contrast": ("BOOLEAN", {"default": True, "tooltip": "contrast [enhancement]"}),
                "contrast_enhancement": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics"
    DESCRIPTION = "Contrast SoX effect node for chaining. dbg-text STRING: 'contrast params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_contrast=True, contrast_enhancement=20.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["contrast", str(contrast_enhancement)]
        debug_str = shlex.join(effect_params)
        if enable_contrast:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxDcshiftNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_dcshift": ("BOOLEAN", {"default": True, "tooltip": "dcshift amount[%]"}),
                "dcshift_amount": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Noise"
    DESCRIPTION = "Dcshift SoX effect node for chaining. dbg-text STRING: 'dcshift params' always (pre-extend; '** Enabled **' prefix if on)."

    def process(self, audio, enable_dcshift=True, dcshift_amount=0.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["dcshift", str(dcshift_amount)]
        debug_str = shlex.join(effect_params)
        if enable_dcshift:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxDeemphNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_deemph": ("BOOLEAN", {"default": True, "tooltip": "deemph [profile]"}),
                "deemph_profile": (["ccir", "50us", "75us", "15khz"], {"default": "ccir"}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Deemph SoX effect node for chaining. dbg-text STRING: 'deemph params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_deemph=True, deemph_profile="ccir", sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["deemph", deemph_profile]
        debug_str = shlex.join(effect_params)
        if enable_deemph:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxDelayNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_delay": ("BOOLEAN", {"default": True, "tooltip": "delay length [pad]"}),
                "delay_length": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 10000.0, "step": 1.0}),
                "delay_pad": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 10000.0, "step": 1.0}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other"
    DESCRIPTION = "Delay SoX effect node for chaining. dbg-text STRING: 'delay params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_delay=True, delay_length=500.0, delay_pad=500.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["delay", str(delay_length), str(delay_pad)]
        debug_str = shlex.join(effect_params)
        if enable_delay:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxDitherNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_dither": ("BOOLEAN", {"default": True, "tooltip": "dither [-s|-a|-h] [n]"}),
                "dither_type": (["s", "a", "h"], {"default": "s"}),
                "dither_depth": ("INT", {"default": 6, "min": 1, "max": 24, "step": 1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Noise"
    DESCRIPTION = "Dither SoX effect node for chaining. dbg-text STRING: 'dither params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_dither=True, dither_type="s", dither_depth=6, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["dither", "-" + dither_type, str(dither_depth)]
        debug_str = shlex.join(effect_params)
        if enable_dither:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxDownsampleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_downsample": ("BOOLEAN", {"default": True, "tooltip": "downsample factor"}),
                "downsample_factor": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other"
    DESCRIPTION = "Downsample SoX effect node for chaining. dbg-text `STRING`: 'downsample params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_downsample=True, downsample_factor=2, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["downsample", str(downsample_factor)]
        debug_str = shlex.join(effect_params)
        if enable_downsample:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxEarwaxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_earwax": ("BOOLEAN", {"default": True, "tooltip": "earwax"}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Modulation"
    DESCRIPTION = "Earwax SoX effect node for chaining. dbg-text `STRING`: 'earwax params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_earwax=True, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["earwax"]
        debug_str = shlex.join(effect_params)
        if enable_earwax:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxFadeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_fade": ("BOOLEAN", {"default": True, "tooltip": "fade [t|h] length [in/out]"}),
                "fade_type": (["h", "t"], {"default": "h"}),
                "fade_in_length": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 60.0, "step": 0.01}),
                "fade_out_length": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 60.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope"
    DESCRIPTION = "Fade SoX effect node for chaining. dbg-text `STRING`: 'fade params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_fade=True, fade_type="h", fade_in_length=0.5, fade_out_length=0.5,
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["fade", fade_type]
        if fade_in_length > 0:
            effect_params.append(str(fade_in_length))
        if fade_out_length > 0:
            effect_params.append(str(fade_out_length))
        debug_str = shlex.join(effect_params)
        if enable_fade:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxFirNode:
    # Tested: DGS v0.1.3
    @classmethod
    def INPUT_TYPES(cls):
        # Get filename from ./fir_coeffs/
        fir_coeffs_dir = "custom_nodes/ComfyUI_SoX_Effects/fir_coeffs/"
        fir_files = []
        if os.path.exists(fir_coeffs_dir) and os.path.isdir(fir_coeffs_dir):
            with os.scandir(fir_coeffs_dir) as entries:
                fir_files = [entry.name for entry in entries if entry.is_file() and entry.name.endswith(".fir")]
        # sort fir_files by name
        fir_files.sort()

        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_fir": ("BOOLEAN", {"default": True, "tooltip": "Enable/Disable SoxFirNode"}),
                "fir_coefficients": (fir_files, {"multiline": True, "default": fir_files[0]}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Fir SoX effect node for chaining (provide coefficients). dbg-text `string`: 'fir params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    # python
    # python
    def process(self, audio, enable_fir=True, fir_coefficients="", sox_params=None):
        fir_coeffs_dir = "custom_nodes/ComfyUI_SoX_Effects/fir_coeffs/"
        fir_file_map = {}
        if os.path.exists(fir_coeffs_dir) and os.path.isdir(fir_coeffs_dir):
            with os.scandir(fir_coeffs_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(".fir"):
                        # store absolute path to ensure full filepath is used later
                        fir_file_map[entry.name] = os.path.abspath(entry.path)

        current_params = sox_params["sox_params"] if sox_params is not None else []

        effect_params = ["fir"] + [fir_file_map[fir_coefficients]]

        debug_str = shlex.join(effect_params)
        if enable_fir and len(effect_params) > 1:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)

        return (audio, {"sox_params": current_params}, debug_str)


class SoxGainNode:
    # Tested: DGS v0.1.3
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "enable_gain": ("BOOLEAN", {"default": True, "tooltip": "Enable gain effect"}),
                "gain_dB": ("FLOAT", {
    "default": 0.0, "min": -18.0, "max": 18.0, "step": 0.1,
    "tooltip": """Primary fixed gain adjustment in dB (appended only if != 0).

Uniformly scales signal amplitude.
• 0 dB: unity (no change)
• Positive: boost volume
• Negative: attenuate

Use early for level staging; pairs with flags like -e/-b."""
}),
                "enable_equalize": ("BOOLEAN", {"default": False, "tooltip": "Enable -e equalize"}),
                "enable_peak": ("BOOLEAN", {"default": False, "tooltip": "Enable -b peak"}),
                "normalize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": """Enable RMS normalization to 0 dB (-n).
    
    Normalizes the audio so the RMS level is 0 dB.
    
    Incompatible with limiter (-l): using both may cause clipping or unexpected results.
    Use normalize_db to set a custom target level (e.g., -12 dB)."""
                }),
                "normalize_db": ("FLOAT", {
                    "default": 0.0,
                    "min": -20.0,
                    "max": 0.0,
                    "step": 0.1,
                    "tooltip": """Custom dB target for normalization (appended to -n if != 0).
    
    Negative values (e.g., -12) set the RMS target below 0 dB for headroom.
    Only applies if normalize is enabled."""
                }),
                "limiter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": """Enable peak limiting to 0 dB (-l).
    
    Applies a soft limiter to prevent peaks from exceeding 0 dB.
    
    Incompatible with normalize (-n): using both may cause clipping or unexpected results.
    Use limiter_db to set a custom threshold (e.g., -6 dB)."""
                }),
                "limiter_db": ("FLOAT", {
                    "default": 0.0,
                    "min": -20.0,
                    "max": 0.0,
                    "step": 0.1,
                    "tooltip": """Custom dB threshold for limiter (appended to -l if != 0).
    
    Negative values (e.g., -6) set the peak limit below 0 dB.
    Only applies if limiter is enabled."""
                }),
                "headroom": ("BOOLEAN", {
                    "default": False,
                    "tooltip": """Enable headroom management (-h).
    
    Automatically calculates and applies gain to use available headroom without clipping.
    
    May interact with reclaim_headroom (-r): use -h before -r for best results.
    Use headroom_db for custom adjustments if needed."""
                }),
                "headroom_db": ("FLOAT", {
                    "default": 0.0,
                    "min": -20.0,
                    "max": 0.0,
                    "step": 0.1,
                    "tooltip": """Custom dB adjustment for headroom (appended to -h if != 0).
    
    Fine-tunes the headroom calculation (negative for more conservative gain).
    Only applies if headroom is enabled."""
                }),
                "reclaim_headroom": ("BOOLEAN", {
                    "default": False,
                    "tooltip": """Enable headroom reclamation (-r).
    
    Boosts the signal after limiting or other processes to reclaim lost headroom.
    
    Best used after limiter (-l) or headroom (-h). May cause clipping if overused."""
                }),
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics"
    DESCRIPTION = "Gain SoX effect node with UI sliders for chaining."

    def process(self, audio, enable_gain=True, gain_dB=0.0, enable_equalize=False, enable_peak=False, normalize=False, normalize_db=0.0, limiter=False, limiter_db=0.0, headroom=False, headroom_db=0.0, reclaim_headroom=False, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params else []
        effect_params = ["gain"]
        if enable_equalize:
            effect_params.append("-e")
        if enable_peak:
            effect_params.append("-b")
        if gain_dB != 0.0:
            effect_params.append(str(gain_dB))
        if normalize:
            effect_params.append("-n")
            if normalize_db != 0.0:
                effect_params.append(str(normalize_db))
        if limiter:
            effect_params.append("-l")
            if limiter_db != 0.0:
                effect_params.append(str(limiter_db))
        if headroom:
            effect_params.append("-h")
            if headroom_db != 0.0:
                effect_params.append(str(headroom_db))
        if reclaim_headroom:
            effect_params.append("-r")
        # Note: Options like -e, -b, -r may be mutually exclusive or cause errors in some combos.
        debug_str = shlex.join(effect_params)
        if enable_gain:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxHilbertNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_hilbert": ("BOOLEAN", {"default": True, "tooltip": "hilbert [window] [halflen]"}),
                "hilbert_window": ("INT", {"default": 64, "min": 8, "max": 1024, "step": 8}),
                "hilbert_halflen": ("INT", {"default": 16, "min": 4, "max": 256, "step": 4}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other"
    DESCRIPTION = "Hilbert SoX effect node for chaining."

    def process(self, audio, enable_hilbert=True, hilbert_window=64, hilbert_halflen=16, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_hilbert:
            effect_params = ["hilbert", str(hilbert_window), str(hilbert_halflen)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxLadspaNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_ladspa": ("BOOLEAN", {"default": True, "tooltip": "ladspa plugin label [params...]"}),
                "ladspa_params": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other"
    DESCRIPTION = "Ladspa SoX effect node for chaining (plugin label params)."

    def process(self, audio, enable_ladspa=True, ladspa_params="", sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_ladspa and ladspa_params.strip():
            params = shlex.split(ladspa_params.strip())
            effect_params = ["ladspa"] + params
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxLoudnessNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_loudness": ("BOOLEAN", {"default": True, "tooltip": "loudness [gain [volume]]"}),
                "loudness_gain": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "loudness_volume": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics"
    DESCRIPTION = "Loudness SoX effect node for chaining."

    def process(self, audio, enable_loudness=True, loudness_gain=4.0, loudness_volume=12.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_loudness:
            effect_params = ["loudness", str(loudness_gain), str(loudness_volume)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxMcompandNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_mcompand": ("BOOLEAN", {"default": True, "tooltip": "mcompand [multi-band compand params]"}),
                "mcompand_params": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics"
    DESCRIPTION = "Mcompand SoX effect node for chaining (multi-band compand params)."

    def process(self, audio, enable_mcompand=True, mcompand_params="", sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_mcompand and mcompand_params.strip():
            params = shlex.split(mcompand_params.strip())
            effect_params = ["mcompand"] + params
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxNoiseprofNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_noiseprof": ("BOOLEAN", {"default": True, "tooltip": "noiseprof [noise.wav]"}),
                "noiseprof_noise_file": ("STRING", {"default": "", "tooltip": "Optional noise profile file"}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Noise"
    DESCRIPTION = "Noiseprof SoX effect node for chaining (generates noise profile)."

    def process(self, audio, enable_noiseprof=True, noiseprof_noise_file="", sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_noiseprof:
            effect_params = ["noiseprof"]
            if noiseprof_noise_file:
                effect_params.append(noiseprof_noise_file)
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxNoiseredNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_noisered": ("BOOLEAN",
                                    {"default": True, "tooltip": "noisered [noise.prof] [amount [precision]]"}),
                "noisered_profile": ("STRING", {"default": "", "tooltip": "noise.prof file"}),
                "noisered_amount": ("FLOAT", {"default": 0.21, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noisered_precision": ("INT", {"default": 4, "min": 0, "max": 6, "step": 1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Noise"
    DESCRIPTION = "Noisered SoX effect node for chaining."

    def process(self, audio, enable_noisered=True, noisered_profile="", noisered_amount=0.21, noisered_precision=4,
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_noisered:
            effect_params = ["noisered"]
            if noisered_profile:
                effect_params.append(noisered_profile)
            effect_params += [str(noisered_amount), str(noisered_precision)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxNormNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_norm": ("BOOLEAN", {"default": True, "tooltip": "norm [-b|-r|-s] [level [precision]]"}),
                "norm_type": (["", "b", "r", "s"], {"default": ""}),
                "norm_level": ("FLOAT", {"default": -3.0, "min": -99.0, "max": 0.0, "step": 0.1}),
                "norm_precision": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics"
    DESCRIPTION = "Norm SoX effect node for chaining."

    def process(self, audio, enable_norm=True, norm_type="", norm_level=-3.0, norm_precision=0.1, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_norm:
            effect_params = ["norm"]
            if norm_type:
                effect_params.append("-" + norm_type)
            effect_params += [str(norm_level), str(norm_precision)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxOopsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_oops": ("BOOLEAN", {"default": True, "tooltip": "oops [threshold]"}),
                "oops_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other"
    DESCRIPTION = "Oops SoX effect node for chaining."

    def process(self, audio, enable_oops=True, oops_threshold=0.8, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_oops:
            effect_params = ["oops", str(oops_threshold)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxPadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_pad": ("BOOLEAN", {"default": True, "tooltip": "pad intro [outro]"}),
                "pad_intro": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.01}),
                "pad_outro": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope"
    DESCRIPTION = "Pad SoX effect node for chaining."

    def process(self, audio, enable_pad=True, pad_intro=0.0, pad_outro=0.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_pad:
            effect_params = ["pad", str(pad_intro), str(pad_outro)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxRateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_rate": ("BOOLEAN", {"default": True, "tooltip": "rate [-v] [-b low [high]] [q|h|v]"}),
                "rate_quality": (["q", "h", "v"], {"default": "q"}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch"
    DESCRIPTION = "Rate SoX effect node for chaining."

    def process(self, audio, enable_rate=True, rate_quality="q", sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_rate:
            effect_params = ["rate", "-v", rate_quality]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxRemixNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_remix": ("BOOLEAN", {"default": True, "tooltip": "remix [-m|--mix|--merge] gains"}),
                "remix_mode": (["", "m", "mix", "merge"], {"default": ""}),
                "remix_gains": ("STRING", {"multiline": True, "default": "1.0"}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Channel"
    DESCRIPTION = "Remix SoX effect node for chaining."

    def process(self, audio, enable_remix=True, remix_mode="", remix_gains="1.0", sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_remix:
            gains = shlex.split(remix_gains.strip())
            effect_params = ["remix"]
            if remix_mode:
                effect_params.append("--" + remix_mode)
            effect_params += gains
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxRepeatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_repeat": ("BOOLEAN", {"default": True, "tooltip": "repeat count"}),
                "repeat_count": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other"
    DESCRIPTION = "Repeat SoX effect node for chaining."

    def process(self, audio, enable_repeat=True, repeat_count=1, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_repeat:
            effect_params = ["repeat", str(repeat_count)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxReverseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_reverse": ("BOOLEAN", {"default": True, "tooltip": "reverse"}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Channel"
    DESCRIPTION = "Reverse SoX effect node for chaining."

    def process(self, audio, enable_reverse=True, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_reverse:
            effect_params = ["reverse"]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxRiaaNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_riaa": ("BOOLEAN", {"default": True, "tooltip": "riaa [pre]"}),
                "riaa_pre": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Riaa SoX effect node for chaining."

    def process(self, audio, enable_riaa=True, riaa_pre=False, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_riaa:
            effect_params = ["riaa"]
            if riaa_pre:
                effect_params.append("pre")
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSilenceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_silence": ("BOOLEAN", {"default": True, "tooltip": "silence [-l] above duration"}),
                "silence_above": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "silence_duration": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 60.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope"
    DESCRIPTION = "Silence SoX effect node for chaining."

    def process(self, audio, enable_silence=True, silence_above=0.01, silence_duration=0.1, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_silence:
            effect_params = ["silence", "1", str(silence_duration), f"{silence_above}%"]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSincNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_sinc": ("BOOLEAN", {"default": True, "tooltip": "sinc [-h] [-n|-t] [-k freq] freq"}),
                "sinc_frequency": ("FLOAT", {"default": 8000.0, "min": 0.0, "max": 96000.0, "step": 100.0}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Sinc SoX effect node for chaining."

    def process(self, audio, enable_sinc=True, sinc_frequency=8000.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_sinc:
            effect_params = ["sinc", str(sinc_frequency)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSpeedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_speed": ("BOOLEAN", {"default": True, "tooltip": "speed factor"}),
                "speed_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch"
    DESCRIPTION = "Speed SoX effect node for chaining."

    def process(self, audio, enable_speed=True, speed_factor=1.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_speed:
            effect_params = ["speed", str(speed_factor)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSpliceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_splice": ("BOOLEAN", {"default": True, "tooltip": "splice start [duration]"}),
                "splice_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.01}),
                "splice_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope"
    DESCRIPTION = "Splice SoX effect node for chaining."

    def process(self, audio, enable_splice=True, splice_start=0.0, splice_duration=1.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_splice:
            effect_params = ["splice", str(splice_start), str(splice_duration)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxStatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_stat": ("BOOLEAN", {"default": True, "tooltip": "stat [tags]"}),
                "stat_tags": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Visualization"
    DESCRIPTION = "Stat SoX effect node for chaining (audio stats)."

    def process(self, audio, enable_stat=True, stat_tags="", sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_stat:
            tags = shlex.split(stat_tags.strip())
            effect_params = ["stat"] + tags
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxStatsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_stats": ("BOOLEAN", {"default": True, "tooltip": "stats [tag]"}),
                "stats_tag": ("STRING", {"default": ""}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Visualization"
    DESCRIPTION = "Stats SoX effect node for chaining."

    def process(self, audio, enable_stats=True, stats_tag="", sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_stats:
            effect_params = ["stats"]
            if stats_tag:
                effect_params.append(stats_tag)
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxStretchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_stretch": ("BOOLEAN", {"default": True, "tooltip": "stretch factor [fadelen]"}),
                "stretch_factor": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01}),
                "stretch_fadelen": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch"
    DESCRIPTION = "Stretch SoX effect node for chaining."

    def process(self, audio, enable_stretch=True, stretch_factor=1.0, stretch_fadelen=0.05, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_stretch:
            effect_params = ["stretch", str(stretch_factor), str(stretch_fadelen)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSwapNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_swap": ("BOOLEAN", {"default": True, "tooltip": "swap [1|2|3|4]"}),
                "swap_operation": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Channel"
    DESCRIPTION = "Swap SoX effect node for chaining."

    def process(self, audio, enable_swap=True, swap_operation=1, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_swap:
            effect_params = ["swap", str(swap_operation)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxSynthNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_synth": ("BOOLEAN", {"default": True, "tooltip": "synth [len] TYPE freq"}),
                "synth_params": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Modulation"
    DESCRIPTION = "Synth SoX effect node for chaining (generator)."

    def process(self, audio, enable_synth=True, synth_params="", sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_synth and synth_params.strip():
            params = shlex.split(synth_params.strip())
            effect_params = ["synth"] + params
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxTrimNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_trim": ("BOOLEAN", {"default": True, "tooltip": "trim start [end]"}),
                "trim_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.01}),
                "trim_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope"
    DESCRIPTION = "Trim SoX effect node for chaining."

    def process(self, audio, enable_trim=True, trim_start=0.0, trim_end=0.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_trim:
            effect_params = ["trim"]
            effect_params.append(str(trim_start))
            if trim_end > 0:
                effect_params.append(str(trim_end))
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxUpsampleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_upsample": ("BOOLEAN", {"default": True, "tooltip": "upsample factor"}),
                "upsample_factor": ("INT", {"default": 2, "min": 1, "max": 32, "step": 1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Other"
    DESCRIPTION = "Upsample SoX effect node for chaining."

    def process(self, audio, enable_upsample=True, upsample_factor=2, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_upsample:
            effect_params = ["upsample", str(upsample_factor)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxVadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_vad": ("BOOLEAN", {"default": True,
                                           "tooltip": "Enable VAD (Voice Activity Detection) SoX effect: Trims silence before/after detected speech/audio activity. Usage: Chain early in workflow → SoxApplyEffectsNode for clean recordings (podcasts, vocals). Pairs with Vol for balance."}),
                "vad_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "VAD threshold (0.0-1.0): Energy level above which audio is considered 'voice'; trims leading/trailing silence. Higher values trim more aggressively."}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Envelope"
    DESCRIPTION = """SoxVadNode: Chains VAD (Voice Activity Detection) SoX effect to SOX_PARAMS.

**What it does**: Adds `vad -t <threshold>` param; trims leading/trailing silence by detecting energy above threshold (0.0-1.0).

**How to use**:
- Toggle `enable_vad`; adjust `vad_threshold` (0.5 default: balanced).
- Wire: AUDIO → SoxVadNode → [Vol/Bass/etc.] → SoxApplyEffectsNode → Output.
- Best early: Clean raw audio (podcasts/vocals) before mixing/gain/EQ.
- Output: Unchanged AUDIO + updated SOX_PARAMS."""

    def process(self, audio, enable_vad=True, vad_threshold=0.5, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_vad:
            effect_params = ["vad", f"-t {vad_threshold}"]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxVolNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_vol": ("BOOLEAN", {"default": True,
                                           "tooltip": "Enable Vol (Volume) SoX effect: Adjusts gain by dB. Usage: Chain for level matching → SoxApplyEffectsNode. Use post-VAD, pre-mix to prevent clipping."}),
                "vol_gain": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 0.1,
                                       "tooltip": "Volume gain in dB: Positive boosts amplitude, negative attenuates. 0dB=unity. Use for per-track balance pre-mix/effects."}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics"
    DESCRIPTION = """SoxVolNode: Chains Vol (Volume Gain) SoX effect to SOX_PARAMS.

**What it does**: Adds `vol <gain_dB>` param; amplifies/attenuates audio linearly in dB (-60/+60 range).

**How to use**:
- Toggle `enable_vol`; set `vol_gain` (0.0=unity; +boost, -=cut).
- Wire: AUDIO → [Vad/etc.] → SoxVolNode → [EQ/Effects] → SoxApplyEffectsNode → Output.
- Best mid-chain: Balance after trim (VAD), before compression/mix. Prevents overload with negative gain.
- Output: Unchanged AUDIO + updated SOX_PARAMS."""

    def process(self, audio, enable_vol=True, vol_gain=0.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_vol:
            effect_params = ["vol", str(vol_gain)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxBassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_bass": ("BOOLEAN", {"default": True,
                                            "tooltip": """bass gain [frequency(100) [width[s|h|k|q|o]](0.5s)]"""}),
                "bass_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "bass_frequency": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "bass_width": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01, }),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Bass SoX effect node for chaining. dbg-text STRING: 'bass params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_bass=True, bass_gain=0.0, bass_frequency=100.0, bass_width=0.5, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["bass", str(bass_gain), str(bass_frequency), str(bass_width)]
        debug_str = shlex.join(effect_params)
        if enable_bass:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxBendNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_bend": ("BOOLEAN", {"default": True,
                                            "tooltip": """bend [-f frame-rate(25)] [-o over-sample(16)] {start,cents,end}"""}),
                "bend_frame_rate": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "bend_over_sample": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "bend_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "bend_cents": ("FLOAT", {"default": 0.0, "min": -1200.0, "max": 1200.0, "step": 1.0}),
                "bend_end_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS", {"tooltip": """Previous SoX effects chain (list of params).
    
    Appends bend params if enabled; passes through for further chaining.
    
    Use: Wire from prior effect nodes (e.g., GainNode → BendNode → ApplyEffectsNode)."""}),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch"
    DESCRIPTION = """Bend SoX effect node for chaining. dbg-text STRING: 'bend params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."""

    def process(self, audio, enable_bend=True, bend_frame_rate=25, bend_over_sample=16, bend_start_time=0.0,
                bend_cents=0.0, bend_end_time=0.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["bend"]
        if bend_frame_rate != 25:
            effect_params += ["-f", str(bend_frame_rate)]
        if bend_over_sample != 16:
            effect_params += ["-o", str(bend_over_sample)]
        effect_params += [str(bend_start_time), str(bend_cents), str(bend_end_time)]
        debug_str = shlex.join(effect_params)
        if enable_bend:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxChorusNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_chorus": ("BOOLEAN", {"default": True,
                                              "tooltip": """chorus gain-in gain-out delay decay speed depth [ -s | -t ]"""}),
                "chorus_gain_in": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_gain_out": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_delay_1": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 2000.0, "step": 1.0}),
                "chorus_decay_1": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_speed_1": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 10.0, "step": 0.01}),
                "chorus_depth_1": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "chorus_shape_1": (["sin", "tri"], {"default": "sin"}),
                "chorus_delay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 1.0}),
                "chorus_decay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_speed_2": ("FLOAT", {"default": 0.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "chorus_depth_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "chorus_shape_2": (["sin", "tri"], {"default": "sin"}),
                "chorus_delay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 1.0}),
                "chorus_decay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_speed_3": ("FLOAT", {"default": 0.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "chorus_depth_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "chorus_shape_3": (["sin", "tri"], {"default": "sin"}),
                "chorus_delay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 1.0}),
                "chorus_decay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chorus_speed_4": ("FLOAT", {"default": 0.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "chorus_depth_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "chorus_shape_4": (["sin", "tri"], {"default": "sin"}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb"
    DESCRIPTION = "Chorus SoX effect node for chaining. dbg-text STRING: 'chorus params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_chorus=True, chorus_gain_in=0.5, chorus_gain_out=0.5,
                chorus_delay_1=40.0, chorus_decay_1=0.8, chorus_speed_1=0.25, chorus_depth_1=2.0, chorus_shape_1="sin",
                chorus_delay_2=0.0, chorus_decay_2=0.0, chorus_speed_2=0.0, chorus_depth_2=0.0, chorus_shape_2="sin",
                chorus_delay_3=0.0, chorus_decay_3=0.0, chorus_speed_3=0.0, chorus_depth_3=0.0, chorus_shape_3="sin",
                chorus_delay_4=0.0, chorus_decay_4=0.0, chorus_speed_4=0.0, chorus_depth_4=0.0, chorus_shape_4="sin",
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["chorus", str(chorus_gain_in), str(chorus_gain_out)]
        taps = []
        for delay, decay, speed, depth, shape in [
            (chorus_delay_1, chorus_decay_1, chorus_speed_1, chorus_depth_1, chorus_shape_1),
            (chorus_delay_2, chorus_decay_2, chorus_speed_2, chorus_depth_2, chorus_shape_2),
            (chorus_delay_3, chorus_decay_3, chorus_speed_3, chorus_depth_3, chorus_shape_3),
            (chorus_delay_4, chorus_decay_4, chorus_speed_4, chorus_depth_4, chorus_shape_4),
        ]:
            if decay > 0.0:
                shape_str = "-s" if shape == "sin" else "-t"
                taps.extend([str(delay), str(decay), str(speed), str(depth), shape_str])
        effect_params += taps
        debug_str = shlex.join(effect_params)
        if enable_chorus:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxCompandNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_compand": ("BOOLEAN", {"default": True, "tooltip": """compand attack1,decay1{,attack2,decay2} [soft-knee-dB:]in-dB1[,out-dB1]{,in-dB2,out-dB2} [gain [initial-volume-dB [delay]]]
        where {} means optional and repeatable and [] means optional.
        dB values are floating point or -inf'; times are in seconds."""}),
                "compand_attack_1": ("FLOAT", {"default": 0.3, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_decay_1": ("FLOAT", {"default": 1.0, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_enable_ad_2": ("BOOLEAN", {"default": False}),
                "compand_attack_2": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_decay_2": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_enable_ad_3": ("BOOLEAN", {"default": False}),
                "compand_attack_3": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_decay_3": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 20.0, "step": 0.01}),
                "compand_enable_soft_knee": ("BOOLEAN", {"default": True}),
                "compand_soft_knee": ("FLOAT", {"default": 6.0, "min": -120.0, "max": 20.0, "step": 0.1}),
                "compand_in_db_1": ("FLOAT", {"default": -70.0, "min": -120.0, "max": 20.0, "step": 1.0}),
                "compand_enable_out_db_1": ("BOOLEAN", {"default": True}),
                "compand_out_db_1": ("FLOAT", {"default": -60.0, "min": -120.0, "max": 20.0, "step": 1.0}),
                "compand_enable_in_out_2": ("BOOLEAN", {"default": True}),
                "compand_in_db_2": ("FLOAT", {"default": -20.0, "min": -120.0, "max": 20.0, "step": 1.0}),
                "compand_out_db_2": ("FLOAT", {"default": -5.0, "min": -120.0, "max": 20.0, "step": 1.0}),
                "compand_enable_gain": ("BOOLEAN", {"default": False}),
                "compand_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "compand_enable_init_vol": ("BOOLEAN", {"default": False}),
                "compand_init_vol": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "compand_enable_delay": ("BOOLEAN", {"default": False}),
                "compand_delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Dynamics"
    DESCRIPTION = "Compand SoX effect node for chaining. dbg-text STRING: 'compand params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_compand=True,
                compand_attack_1=0.3, compand_decay_1=1.0,
                compand_enable_ad_2=False,
                compand_attack_2=0.0, compand_decay_2=0.0,
                compand_enable_ad_3=False,
                compand_attack_3=0.0, compand_decay_3=0.0,
                compand_enable_soft_knee=True,
                compand_soft_knee=6.0,
                compand_in_db_1=-70.0,
                compand_enable_out_db_1=True,
                compand_out_db_1=-60.0,
                compand_enable_in_out_2=True,
                compand_in_db_2=-20.0,
                compand_out_db_2=-5.0,
                compand_enable_gain=False,
                compand_gain=0.0,
                compand_enable_init_vol=False,
                compand_init_vol=0.0,
                compand_enable_delay=False,
                compand_delay=0.0,
                sox_params=None):
        # Clamp time parameters to >= 0.0 for valid SoX args
        compand_attack_1 = max(0.0, compand_attack_1)
        compand_decay_1 = max(0.0, compand_decay_1)
        compand_attack_2 = max(0.0, compand_attack_2)
        compand_decay_2 = max(0.0, compand_decay_2)
        compand_attack_3 = max(0.0, compand_attack_3)
        compand_decay_3 = max(0.0, compand_decay_3)
        compand_soft_knee = max(0.0, compand_soft_knee)
        compand_delay = max(0.0, compand_delay)
        current_params = sox_params["sox_params"] if sox_params is not None else []
        debug_str = "Compand disabled"
        if enable_compand:
            attack_decay_parts = [f"{compand_attack_1},{compand_decay_1}"]
            if compand_enable_ad_2:
                attack_decay_parts.append(f"{compand_attack_2},{compand_decay_2}")
            if compand_enable_ad_3:
                attack_decay_parts.append(f"{compand_attack_3},{compand_decay_3}")
            attack_decay = " ".join(attack_decay_parts)
            transfer_parts = [str(compand_in_db_1)]
            if compand_enable_out_db_1:
                transfer_parts[0] += f",{compand_out_db_1}"
            if compand_enable_in_out_2:
                transfer_parts.append(f"{compand_in_db_2},{compand_out_db_2}")
            transfer_str = " ".join(transfer_parts)
            if not transfer_parts:
                debug_str = "Compand skipped: no transfer pairs defined"
            else:
                if compand_enable_soft_knee:
                    knee_str = str(compand_soft_knee)
                    compand_str = f"{attack_decay} {knee_str}:{transfer_str}"
                else:
                    compand_str = f"{attack_decay} {transfer_str}"
                tail_parts = []
                if compand_enable_gain:
                    tail_parts.append(str(compand_gain))
                    if compand_enable_init_vol:
                        tail_parts.append(str(compand_init_vol))
                    if compand_enable_delay:
                        tail_parts.append(str(compand_delay))
                if tail_parts:
                    compand_str += " " + " ".join(tail_parts)
                compand_str = compand_str.strip()
                debug_str = compand_str
                effect_params = ["compand"] + shlex.split(compand_str)
                current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxEchoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_echo": ("BOOLEAN", {"default": True,
                                            "tooltip": """echo gain-in gain-out delay decay [ delay decay ... ]"""}),
                "echo_gain_in": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_gain_out": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay_1": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echo_decay_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echo_decay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echo_decay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echo_decay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb"
    DESCRIPTION = "Echo SoX effect node for chaining. dbg-text STRING: 'echo gain-in gain-out [delay decay ...]' always (pre-extend, survives disable)."

    def process(self, audio, enable_echo=True, echo_gain_in=0.8, echo_gain_out=0.9,
                echo_delay_1=1000.0, echo_decay_1=0.5,
                echo_delay_2=0.0, echo_decay_2=0.0,
                echo_delay_3=0.0, echo_decay_3=0.0,
                echo_delay_4=0.0, echo_decay_4=0.0,
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        # Compute debug always
        taps = []
        for delay, decay in [(echo_delay_1, echo_decay_1),
                             (echo_delay_2, echo_decay_2),
                             (echo_delay_3, echo_decay_3),
                             (echo_delay_4, echo_decay_4)]:
            if decay > 0.0:
                taps.extend([str(delay), str(decay)])
        debug_str = shlex.join(["echo", str(echo_gain_in), str(echo_gain_out)] + taps)
        if enable_echo:
            debug_str = "** Enabled **\n" + debug_str
            if taps:
                current_params.extend(["echo", str(echo_gain_in), str(echo_gain_out)] + taps)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxEchosNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_echos": ("BOOLEAN", {"default": True,
                                             "tooltip": """echos gain-in gain-out delay decay [ delay decay ... ]"""}),
                "echos_gain_in": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_gain_out": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_delay_1": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echos_decay_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_delay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echos_decay_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_delay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echos_decay_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echos_delay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "echos_decay_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb"
    DESCRIPTION = "Echos SoX effect node for chaining. dbg-text STRING: 'echos params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_echos=True, echos_gain_in=0.8, echos_gain_out=0.9,
                echos_delay_1=1000.0, echos_decay_1=0.5,
                echos_delay_2=0.0, echos_decay_2=0.0,
                echos_delay_3=0.0, echos_decay_3=0.0,
                echos_delay_4=0.0, echos_decay_4=0.0,
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        taps = []
        for delay, decay in [(echos_delay_1, echos_decay_1),
                             (echos_delay_2, echos_decay_2),
                             (echos_delay_3, echos_decay_3),
                             (echos_delay_4, echos_decay_4)]:
            if decay > 0.0:
                taps.extend([str(delay), str(decay)])
        effect_params = ["echos", str(echos_gain_in), str(echos_gain_out)] + taps
        debug_str = shlex.join(effect_params)
        if enable_echos and taps:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        elif enable_echos:
            debug_str = "** Enabled **\n" + debug_str
        return (audio, {"sox_params": current_params}, debug_str)


class SoxEqualizerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_equalizer": ("BOOLEAN",
                                     {"default": True, "tooltip": """equalizer frequency width[q|o|h|k] gain"""}),
                "equalizer_frequency": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "equalizer_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "equalizer_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Equalizer SoX effect node for chaining."

    def process(self, audio, enable_equalizer=True, equalizer_frequency=1000.0, equalizer_width=1.0, equalizer_gain=0.0,
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_equalizer:
            effect_params = ["equalizer", str(equalizer_frequency), str(equalizer_width) + "q", str(equalizer_gain)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxFlangerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_flanger": ("BOOLEAN", {"default": True, "tooltip": """flanger [delay depth regen width speed shape phase interp]
                  .
                 /|regen
                / |
            +--(  |------------+
            |   \\ |            |   .
           _V_   \\|  _______   |   |\\ width   ___
          |   |   ' |       |  |   | \\       |   |
      +-->| + |---->| DELAY |--+-->|  )----->|   |
      |   |___|     |_______|      | /       |   |
      |           delay : depth    |/        |   |
  In  |                 : interp   '         |   | Out
  --->+               __:__                  | + |--->
      |              |     |speed            |   |
      |              |  ~  |shape            |   |
      |              |_____|phase            |   |
      +------------------------------------->|   |
                                             |___|
       RANGE DEFAULT DESCRIPTION
delay   0 30    0    base delay in milliseconds
depth   0 10    2    added swept delay in milliseconds
regen -95 +95   0    percentage regeneration (delayed signal feedback)
width   0 100   71   percentage of delayed signal mixed with original
speed  0.1 10  0.5   sweeps per second (Hz)
shape    --    sin   swept wave shape: sine|triangle
phase   0 100   25   swept wave percentage phase-shift for multi-channel
                     (e.g. stereo) flange; 0 = 100 = same phase on each channel
interp   --    lin   delay-line interpolation: linear|quadratic"""}),
                "flanger_delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "flanger_depth": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "flanger_regen": ("FLOAT", {"default": 0.0, "min": -95.0, "max": 95.0, "step": 1.0}),
                "flanger_width": ("FLOAT", {"default": 71.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "flanger_speed": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "flanger_shape": (["sinusoidal", "triangular"], {"default": "sinusoidal"}),
                "flanger_phase": ("INT", {"default": 25, "min": 0, "max": 100, "step": 1}),
                "flanger_interp": (["linear", "quadratic"], {"default": "linear"}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb"
    DESCRIPTION = "Flanger SoX effect node for chaining. dbg-text `string`: 'flanger params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_flanger=True, flanger_delay=0.0, flanger_depth=2.0, flanger_regen=0.0,
                flanger_width=71.0, flanger_speed=0.5, flanger_shape="sinusoidal", flanger_phase=25,
                flanger_interp="linear", sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        shape_str = "sin" if flanger_shape == "sinusoidal" else "tri"
        interp_str = "lin" if flanger_interp == "linear" else "quad"
        effect_params = ["flanger",
                         str(flanger_delay), str(flanger_depth), str(flanger_regen),
                         str(flanger_width), str(flanger_speed), shape_str, str(flanger_phase), interp_str
                         ]
        debug_str = shlex.join(effect_params)
        if enable_flanger:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxHighpassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_highpass": ("BOOLEAN", {"default": True,
                                                "tooltip": """highpass [-1|-2] frequency [width[q|o|h|k](0.707q)]"""}),
                "highpass_poles": ("INT", {"default": 2, "min": 1, "max": 2, "step": 1}),
                "highpass_frequency": ("FLOAT", {"default": 3000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "highpass_width": ("FLOAT", {"default": 0.707, "min": 0.1, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Highpass SoX effect node for chaining."

    def process(self, audio, enable_highpass=True, highpass_poles=2, highpass_frequency=3000.0, highpass_width=0.707,
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_highpass:
            poles = "-1" if highpass_poles == 1 else "-2"
            effect_params = ["highpass", poles, str(highpass_frequency), str(highpass_width) + "q"]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxLowpassNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_lowpass": ("BOOLEAN", {"default": True,
                                               "tooltip": """lowpass [-1|-2] frequency [width[q|o|h|k]](0.707q)"""}),
                "lowpass_poles": ("INT", {"default": 2, "min": 1, "max": 2, "step": 1}),
                "lowpass_frequency": ("FLOAT", {"default": 1000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "lowpass_width": ("FLOAT", {"default": 0.707, "min": 0.1, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Lowpass SoX effect node for chaining."

    def process(self, audio, enable_lowpass=True, lowpass_poles=2, lowpass_frequency=1000.0, lowpass_width=0.707,
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_lowpass:
            poles = "-1" if lowpass_poles == 1 else "-2"
            effect_params = ["lowpass", poles, str(lowpass_frequency), str(lowpass_width) + "q"]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxOverdriveNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_overdrive": ("BOOLEAN", {"default": True, "tooltip": """overdrive [gain [colour]]"""}),
                "overdrive_gain": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "overdrive_colour": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb"
    DESCRIPTION = "Overdrive SoX effect node for chaining."

    def process(self, audio, enable_overdrive=True, overdrive_gain=20.0, overdrive_colour=20.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_overdrive:
            effect_params = ["overdrive", str(overdrive_gain), str(overdrive_colour)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxPhaserNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_phaser": ("BOOLEAN", {"default": True,
                                              "tooltip": """phaser gain-in gain-out delay decay speed [ -s | -t ]"""}),
                "phaser_gain_in": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "phaser_gain_out": ("FLOAT", {"default": 0.74, "min": 0.0, "max": 1.0, "step": 0.01}),
                "phaser_delay": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "phaser_decay": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 0.5, "step": 0.01}),
                "phaser_speed": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
                "phaser_mod": (["sinusoidal", "triangular"], {"default": "sinusoidal"}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb"
    DESCRIPTION = "Phaser SoX effect node for chaining."

    def process(self, audio, enable_phaser=True, phaser_gain_in=0.8, phaser_gain_out=0.74, phaser_delay=3.0,
                phaser_decay=0.4, phaser_speed=0.5, phaser_mod="sinusoidal", sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_phaser:
            mod = "-s" if phaser_mod == "sinusoidal" else "-t"
            effect_params = ["phaser",
                             str(phaser_gain_in), str(phaser_gain_out),
                             str(phaser_delay), str(phaser_decay),
                             str(phaser_speed), mod
                             ]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxPitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_pitch": ("BOOLEAN", {"default": True,
                                             "tooltip": """pitch [-q] shift-in-cents [segment-ms [search-ms [overlap-ms]]]"""}),
                "pitch_q": ("BOOLEAN", {"default": False}),
                "pitch_shift": ("INT", {"default": 0, "min": -1200, "max": 1200, "step": 1}),
                "pitch_segment": ("FLOAT", {"default": 82.0, "min": 10.0, "max": 200.0, "step": 1.0}),
                "pitch_search": ("FLOAT", {"default": 14.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "pitch_overlap": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 50.0, "step": 1.0}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch"
    DESCRIPTION = "Pitch SoX effect node for chaining."

    def process(self, audio, enable_pitch=True, pitch_q=False, pitch_shift=0, pitch_segment=82.0, pitch_search=14.0,
                pitch_overlap=12.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_pitch:
            effect_params = ["pitch"]
            if pitch_q:
                effect_params += ["-q"]
            effect_params += [str(pitch_shift)]
            if pitch_segment != 82.0 or pitch_search != 14.0 or pitch_overlap != 12.0:
                effect_params += [str(pitch_segment), str(pitch_search), str(pitch_overlap)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxReverbNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_reverb": ("BOOLEAN", {"default": True,
                                              "tooltip": """reverb [-w|--wet-only] [reverberance (50%) [HF-damping (50%) [room-scale (100%) [stereo-depth (100%) [pre-delay (0ms) [wet-gain (0dB)]]]]]]"""}),
                "reverb_wet_only": ("BOOLEAN", {"default": False}),
                "reverb_reverberance": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "reverb_hf_damping": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "reverb_room_scale": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "reverb_stereo_depth": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "reverb_pre_delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 500.0, "step": 1.0}),
                "reverb_wet_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Reverb"
    DESCRIPTION = "Reverb SoX effect node for chaining."

    def process(self, audio, enable_reverb=True, reverb_wet_only=False, reverb_reverberance=50.0,
                reverb_hf_damping=50.0, reverb_room_scale=100.0, reverb_stereo_depth=100.0, reverb_pre_delay=0.0,
                reverb_wet_gain=0.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_reverb:
            effect_params = ["reverb"]
            if reverb_wet_only:
                effect_params += ["--wet-only"]
            effect_params += [
                str(reverb_reverberance), str(reverb_hf_damping),
                str(reverb_room_scale), str(reverb_stereo_depth),
                str(reverb_pre_delay), str(reverb_wet_gain)
            ]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxTempoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_tempo": ("BOOLEAN", {"default": True,
                                             "tooltip": """tempo [-q] [-m | -s | -l] factor [segment-ms [search-ms [overlap-ms]]]"""}),
                "tempo_q": ("BOOLEAN", {"default": False}),
                "tempo_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "tempo_segment": ("FLOAT", {"default": 82.0, "min": 10.0, "max": 200.0, "step": 1.0}),
                "tempo_search": ("FLOAT", {"default": 14.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "tempo_overlap": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 50.0, "step": 1.0}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Pitch"
    DESCRIPTION = "Tempo SoX effect node for chaining."

    def process(self, audio, enable_tempo=True, tempo_q=False, tempo_factor=1.0, tempo_segment=82.0, tempo_search=14.0,
                tempo_overlap=12.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        if enable_tempo:
            effect_params = ["tempo"]
            if tempo_q:
                effect_params += ["-q"]
            effect_params += [str(tempo_factor)]
            if tempo_segment != 82.0 or tempo_search != 14.0 or tempo_overlap != 12.0:
                effect_params += [str(tempo_segment), str(tempo_search), str(tempo_overlap)]
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params})


class SoxTrebleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_treble": ("BOOLEAN", {"default": True,
                                              "tooltip": """treble gain [frequency(3000) [width[s|h|k|q|o]](0.5s)]"""}),
                "treble_gain": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "treble_frequency": ("FLOAT", {"default": 3000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "treble_width": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Equalization"
    DESCRIPTION = "Treble SoX effect node for chaining. dbg-text `string`: 'treble params' always (pre-extend; '** Enabled **' prefix if on). Wire to PreviewTextNode."

    def process(self, audio, enable_treble=True, treble_gain=0.0, treble_frequency=3000.0, treble_width=0.5,
                sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["treble", str(treble_gain), str(treble_frequency), str(treble_width)]
        debug_str = shlex.join(effect_params)
        if enable_treble:
            debug_str = "** Enabled **\n" + debug_str
            current_params.extend(effect_params)
        return (audio, {"sox_params": current_params}, debug_str)


class SoxTremoloNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "enable_tremolo": ("BOOLEAN", {"default": True, "tooltip": """tremolo speed_Hz [depth_percent]"""}),
                "tremolo_speed": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "tremolo_depth": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "sox_params": ("SOX_PARAMS",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SOX_PARAMS", "STRING")
    RETURN_NAMES = ("audio", "sox_params", "dbg-text")
    FUNCTION = "process"
    CATEGORY = "audio/SoX/Effects/Modulation"
    DESCRIPTION = "Tremolo SoX effect node for chaining."

    def process(self, audio, enable_tremolo=True, tremolo_speed=0.5, tremolo_depth=40.0, sox_params=None):
        current_params = sox_params["sox_params"] if sox_params is not None else []
        effect_params = ["tremolo", str(tremolo_speed), str(tremolo_depth)]
        cmd_str = f"sox voice.wav vibrato.wav {' '.join(effect_params)}"
        dbg_text = cmd_str
        if enable_tremolo:
            dbg_text = "** Enabled **\n" + cmd_str
            current_params.extend(effect_params)
        else:
            dbg_text = "tremolo disabled"
        return (audio, {"sox_params": current_params}, dbg_text)

NODE_CLASS_MAPPINGS = {
    "SoxApplyEffects": SoxApplyEffectsNode,
    "SoxAllpass": SoxAllpassNode,
    "SoxBand": SoxBandNode,
    "SoxBandpass": SoxBandpassNode,
    "SoxBandreject": SoxBandrejectNode,
    "SoxBass": SoxBassNode,
    "SoxBend": SoxBendNode,
    "SoxBiquad": SoxBiquadNode,
    "SoxChannels": SoxChannelsNode,
    "SoxChorus": SoxChorusNode,
    "SoxCompand": SoxCompandNode,
    "SoxContrast": SoxContrastNode,
    "SoxDcshift": SoxDcshiftNode,
    "SoxDeemph": SoxDeemphNode,
    "SoxDelay": SoxDelayNode,
    "SoxDither": SoxDitherNode,
    "SoxDownsample": SoxDownsampleNode,
    "SoxEarwax": SoxEarwaxNode,
    "SoxEcho": SoxEchoNode,
    "SoxEchos": SoxEchosNode,
    "SoxEqualizer": SoxEqualizerNode,
    "SoxFade": SoxFadeNode,
    "SoxFir": SoxFirNode,
    "SoxFlanger": SoxFlangerNode,
    "SoxGain": SoxGainNode,
    "SoxHighpass": SoxHighpassNode,
    "SoxHilbert": SoxHilbertNode,
    "SoxLadspa": SoxLadspaNode,
    "SoxLoudness": SoxLoudnessNode,
    "SoxLowpass": SoxLowpassNode,
    "SoxMcompand": SoxMcompandNode,
    "SoxNoiseprof": SoxNoiseprofNode,
    "SoxNoisered": SoxNoiseredNode,
    "SoxNorm": SoxNormNode,
    "SoxOops": SoxOopsNode,
    "SoxOverdrive": SoxOverdriveNode,
    "SoxPad": SoxPadNode,
    "SoxPhaser": SoxPhaserNode,
    "SoxPitch": SoxPitchNode,
    "SoxRate": SoxRateNode,
    "SoxRemix": SoxRemixNode,
    "SoxRepeat": SoxRepeatNode,
    "SoxReverb": SoxReverbNode,
    "SoxReverse": SoxReverseNode,
    "SoxRiaa": SoxRiaaNode,
    "SoxSilence": SoxSilenceNode,
    "SoxSinc": SoxSincNode,
    "SoxSpeed": SoxSpeedNode,
    "SoxSplice": SoxSpliceNode,
    "SoxStat": SoxStatNode,
    "SoxStats": SoxStatsNode,
    "SoxStretch": SoxStretchNode,
    "SoxSwap": SoxSwapNode,
    "SoxSynth": SoxSynthNode,
    "SoxTempo": SoxTempoNode,
    "SoxTreble": SoxTrebleNode,
    "SoxTremolo": SoxTremoloNode,
    "SoxTrim": SoxTrimNode,
    "SoxUpsample": SoxUpsampleNode,
    "SoxUtilSpectrogram": "Sox Spectrogram",
    "SoxVad": SoxVadNode,
    "SoxVol": SoxVolNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoxApplyEffects": "SoX Apply Effects",
    "SoxAllpass": "SoX Allpass",
    "SoxBand": "SoX Band",
    "SoxBandpass": "SoX Bandpass",
    "SoxBandreject": "SoX Bandreject",
    "SoxBass": "SoX Bass",
    "SoxBend": "SoX Bend",
    "SoxBiquad": "SoX Biquad",
    "SoxChannels": "SoX Channels",
    "SoxChorus": "SoX Chorus",
    "SoxCompand": "SoX Compand",
    "SoxContrast": "SoX Contrast",
    "SoxDcshift": "SoX Dcshift",
    "SoxDeemph": "SoX Deemph",
    "SoxDelay": "SoX Delay",
    "SoxDither": "SoX Dither",
    "SoxDownsample": "SoX Downsample",
    "SoxEarwax": "SoX Earwax",
    "SoxEcho": "SoX Echo",
    "SoxEchos": "SoX Echos",
    "SoxEqualizer": "SoX Equalizer",
    "SoxFade": "SoX Fade",
    "SoxFir": "SoX Fir",
    "SoxFlanger": "SoX Flanger",
    "SoxGain": "SoX Gain",
    "SoxHighpass": "SoX Highpass",
    "SoxHilbert": "SoX Hilbert",
    "SoxLadspa": "SoX Ladspa",
    "SoxLoudness": "SoX Loudness",
    "SoxLowpass": "SoX Lowpass",
    "SoxMcompand": "SoX Mcompand",
    "SoxNoiseprof": "SoX Noiseprof",
    "SoxNoisered": "SoX Noisered",
    "SoxNorm": "SoX Norm",
    "SoxOops": "SoX Oops",
    "SoxOverdrive": "SoX Overdrive",
    "SoxPad": "SoX Pad",
    "SoxPhaser": "SoX Phaser",
    "SoxPitch": "SoX Pitch",
    "SoxRate": "SoX Rate",
    "SoxRemix": "SoX Remix",
    "SoxRepeat": "SoX Repeat",
    "SoxReverb": "SoX Reverb",
    "SoxReverse": "SoX Reverse",
    "SoxRiaa": "SoX Riaa",
    "SoxSilence": "SoX Silence",
    "SoxSinc": "SoX Sinc",
    "SoxSpeed": "SoX Speed",
    "SoxSplice": "SoX Splice",
    "SoxStat": "SoX Stat",
    "SoxStats": "SoX Stats",
    "SoxStretch": "SoX Stretch",
    "SoxSwap": "SoX Swap",
    "SoxSynth": "SoX Synth",
    "SoxTempo": "SoX Tempo",
    "SoxTreble": "SoX Treble",
    "SoxTremolo": "SoX Tremolo",
    "SoxTrim": "SoX Trim",
    "SoxUpsample": "SoX Upsample",
    "SoxUtilSpectrogram": "SoX Spectrogram",
    "SoxVad": "SoX Vad",
    "SoxVol": "SoX Vol",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
