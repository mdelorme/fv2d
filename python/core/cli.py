import argparse
from matplotlib import colormaps
from fv2d_utils import latexify

# TODO: create a parent parser to hold all similar args
class PlotCLI:
    """Classe pour gérer la ligne de commande et les sous-commandes."""

    def __init__(self):
        self.parser = self._create_parser()
        self.subparsers = self.parser.add_subparsers(dest='command', required=True, help="Sous-commandes disponibles.")
        self._add_field_subcommand()
        self._add_slice_subcommand()
        self._add_compare_subcommand()

    def _create_parser(self):
        """Crée le parser principal avec l'option commune `--file`."""
        parser = argparse.ArgumentParser(description="Outil de traçage pour simulations.")
        return parser

    def _add_field_subcommand(self):
        """Ajoute la sous-commande `field` (champs 2D)."""
        field_parser = self.subparsers.add_parser('field', help="Trace des champs 2D.")
        field_parser.add_argument(
            "-f", "--file",
            nargs='+',
            required=True,
            help="Chemin vers le(s) fichier(s) .h5 (ou motif comme 'simulation_*.h5')."
        )
        field_parser.add_argument(
            "-t", "--field",
            choices=latexify.keys(),
            required=True,
            help="Champ à tracer (ex: 'rho', 'bx')."
        )
        field_parser.add_argument(
            "--solver",
            help="Solver associated with the `.h5` file(s)."
        )
        field_parser.add_argument(
            "--colormap",
            choices=list(colormaps),
            default="plasma",
            help="Colormap à utiliser."
        )
        field_parser.add_argument(
            "--show-grid",
            action="store_true",
            help="Affiche une grille sur les plots."
        )
        field_parser.add_argument(
            "--flipy",
            action="store_true",
            help="Inverse l'axe y."
        )
        field_parser.add_argument(
            "--save-mp4",
            action="store_true",
            help="Sauvegarde une vidéo MP4."
        )
        field_parser.add_argument(
            "--fps",
            type=int,
            default=25,
            help="FPS pour la vidéo."
        )
        field_parser.add_argument(
            "--quiverB",
            action="store_true",
            help="Add a quiver plot of the magnetic vector field."
        )
        field_parser.add_argument(
            "--streamplotV",
            action="store_true",
            help="Add a streamplot of the velocity vector field."
        )
       
        field_parser.add_argument(
            "--streamplotB",
            action="store_true",
            help="Add a streamplot of the magnetic vector field."
        )
        

        field_parser.add_argument(
            "-b", "--boundaries",
            nargs=2,
            help="Minimum and maximum values to display on the image."
        )

        contours_group = field_parser.add_argument_group("contours", "Add contours of the specified field.")
        contours_group.add_argument(
            "--contours",
            choices=latexify.keys(),
            default=None,
            help="Add contours of the specified field."
        )

        contours_group.add_argument(
            "-l", "--levels",
            nargs='*',
            default=5,
            help="Level values for the contour plot."
        )


    def _add_slice_subcommand(self):
        """Ajoute la sous-commande `slice` (slices 1D)."""
        slice_parser = self.subparsers.add_parser('slice', help="Trace des slices 1D.")
        slice_parser.add_argument(
            "-f", "--file",
            nargs='+',
            required=True,
            help="Chemin vers le(s) fichier(s) .h5 (ou motif comme 'simulation_*.h5')."
        )
        slice_parser.add_argument(
            "-t", "--field",
            choices=latexify.keys(),
            default="rho",
            help="Champ à tracer."
        )
        slice_group = slice_parser.add_mutually_exclusive_group(required=True)
        slice_group.add_argument(
            "-y", "--yslice",
            type=float,
            help="Valeur de y pour la slice horizontale (ex: y=0.5)."
        )
        slice_group.add_argument(
            "-x", "--xslice",
            type=float,
            help="Valeur de x pour la slice verticale (ex: x=1.0)."
        )
        slice_parser.add_argument(
            "--solver",
            help="Solver associated with the `.h5` file(s)."
        )
        slice_parser.add_argument(
            "--colormap",
            choices=list(colormaps),
            default="plasma",
            help="Colormap à utiliser."
        )
        slice_parser.add_argument(
            "--show-grid",
            action="store_true",
            help="Affiche une grille sur les plots."
        )
        slice_parser.add_argument(
            "--flipy",
            action="store_true",
            help="Inverse l'axe y."
        )
        slice_parser.add_argument(
            "--save-mp4",
            action="store_true",
            help="Sauvegarde une vidéo MP4."
        )
        slice_parser.add_argument(
            "--fps",
            type=int,
            default=25,
            help="FPS pour la vidéo."
        )

    def _add_compare_subcommand(self):
        """Ajoute la sous-commande `compare` (comparaisons)."""
        compare_parser = self.subparsers.add_parser('compare', help="Compare des champs.")
        compare_parser.add_argument(
            "-f", "--file",
            nargs='+',
            required=True,
            help="Chemin vers le(s) fichier(s) .h5 (ou motif comme 'simulation_*.h5')."
        )
        compare_parser.add_argument(
            "-t", "--field",
            choices=latexify.keys(),
            default="rho",
            help="Champ à comparer (si --fields n'est pas spécifié)."
        )
        compare_parser.add_argument(
            "--fields",
            nargs='+',
            default=None,
            help="Liste des champs à comparer (ex: 'Bmag v')."
        )
        compare_parser.add_argument(
            "--mode",
            choices=["side-by-side", "overlay"],
            default="side-by-side",
            help="Mode de comparaison."
        )
        compare_parser.add_argument(
            "--labels",
            nargs='+',
            help="Labels pour chaque champ/simulation."
        )
        compare_parser.add_argument(
            "--solver",
            help="Solver associated with the `.h5` file(s)."
        )
        compare_parser.add_argument(
            "--colormap",
            choices=list(colormaps),
            default="plasma",
            help="Colormap à utiliser."
        )
        compare_parser.add_argument(
            "--show-grid",
            action="store_true",
            help="Affiche une grille sur les plots."
        )
        compare_parser.add_argument(
            "--flipy",
            action="store_true",
            help="Inverse l'axe y."
        )
        compare_parser.add_argument(
            "--save-mp4",
            action="store_true",
            help="Sauvegarde une vidéo MP4."
        )
        compare_parser.add_argument(
            "--fps",
            type=int,
            default=25,
            help="FPS pour la vidéo."
        )

    def parse_args(self):
        """Parse les arguments de la ligne de commande."""
        return self.parser.parse_args()
