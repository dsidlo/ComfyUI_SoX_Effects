#
# Contains a utility functions class containing static methods
# that are commonly used across ComfyUI_Sox_Effects Nodes.
#
import subprocess
import os

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
            y=240
    ):
        """
        Renders a SoX --plot gnuplot script to a PNG file.

        Requires gnuplot installed on your system.

        # Example usage after you ran:
        # sox --plot gnuplot -n -n fir coeffs.txt > fir_plot.gp
        # render_sox_plot_to_image('fir_plot.gp', 'fir_response.png', x=1200, y=700)
        """
        if not os.path.exists(sox_plot_script_path):
            raise FileNotFoundError(f"Script not found: {sox_plot_script_path}")

        # Commands to force PNG output and avoid interactive window
        commands = [
            "set terminal pngcairo size {x},{y} enhanced font 'Arial,10'".format(x=x, y=y),
            f"set output '{output_image}'",
            f"load '{sox_plot_script_path}'",
            "unset output",
            "exit"  # Ensures gnuplot terminates cleanly
        ]

        gnuplot_input = '\n'.join(commands) + '\n'

        try:
            result = subprocess.run(
                ['gnuplot', '-e', gnuplot_input],  # -e executes commands directly
                capture_output=True,
                text=True,
                check=True
            )
            print(f"PNG successfully generated: {os.path.abspath(output_image)}")
            if result.stdout:
                print("gnuplot stdout:", result.stdout)
            if result.stderr:
                print("gnuplot warnings/errors:", result.stderr)
        except subprocess.CalledProcessError as e:
            print("Error running gnuplot:")
            print(e.stderr)
            raise

