#
# Contains a utility functions class containing static methods
# that are commonly used across ComfyUI_Sox_Effects Nodes.
#
import subprocess
import os
import shutil

class SoxNodeUtils:
    """
    Static methods used by ComfyUI_Sox_Effects Nodes
    - This class should not be added to "NODE_CLASS_MAPPINGS" and/or "NODE_DISPLAY_NAME_MAPPINGS" data structures.
    """

    @staticmethod
    def render_sox_plot_to_image(
            sox_plot_script_path,
            output_image='transfer_function.png',
            x=800,
            y=400
    ):
        """
        Renders a SoX --plot gnuplot script to a PNG file.

        Requires gnuplot installed on your system.

        Returns (response_msg, stdout, stderr):
        - response_msg: None on success, error message string otherwise
        - stdout, stderr: from gnuplot subprocess

        # Example usage after you ran:
        # sox --plot gnuplot -n -n fir coeffs.txt > fir_plot.gp
        # msg, out, err = render_sox_plot_to_image('fir_plot.gp', 'fir_response.png', x=1200, y=700)
        """
        if not os.path.exists(sox_plot_script_path):
            return (f"Script not found: {sox_plot_script_path}", "", "")

        if shutil.which('gnuplot') is None:
            return ("Gnuplot command not found. Please install gnuplot and ensure it's in your PATH.", "", "")

        # Commands to force PNG output and avoid interactive window
        commands = [
            f"set terminal pngcairo size {x},{y} enhanced font 'Arial,10'",
            f"set output '{output_image}'",
            f"load '{sox_plot_script_path}'",
            "unset output",
            "exit"  # Ensures gnuplot terminates cleanly
        ]

        gnuplot_input = '\n'.join(commands) + '\n'

        result = subprocess.run(
            ['gnuplot', '-e', gnuplot_input],  # -e executes commands directly
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            msg = f"Gnuplot failed (rc={result.returncode})."
            if result.stderr.strip():
                msg += f" STDERR: {result.stderr.strip()}"
            print("Error running gnuplot:")
            print(result.stderr)
            return (msg, result.stdout, result.stderr)

        print(f"PNG successfully generated: {os.path.abspath(output_image)}")
        if result.stdout.strip():
            print("gnuplot stdout:", result.stdout)
        if result.stderr.strip():
            print("gnuplot warnings/errors:", result.stderr)

        return (None, result.stdout, result.stderr)

