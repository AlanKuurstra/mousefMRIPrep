from nipype.interfaces.ants.base import ANTSCommand, ANTSCommandInputSpec
from nipype.interfaces.base import (TraitedSpec, File, traits,  Str)
from .fixheader import CopyHeaderInterface

class ImageMathInputSpec(ANTSCommandInputSpec):
    dimension = traits.Int(
        3, usedefault=True, position=1, argstr="%d", desc="dimension of output image"
    )
    output_image = File(
        position=2,
        argstr="%s",
        name_source=["op1"],
        name_template="%s_maths",
        desc="output image file",
        keep_extension=True,
    )
    operation = traits.Enum(
        "m",
        "vm",
        "+",
        "v+",
        "-",
        "v-",
        "/",
        "^",
        "max",
        "exp",
        "addtozero",
        "overadd",
        "abs",
        "total",
        "mean",
        "vtotal",
        "Decision",
        "Neg",
        "Project",
        "G",
        "MD",
        "ME",
        "MO",
        "MC",
        "GD",
        "GE",
        "GO",
        "GC",
        mandatory=True,
        position=3,
        argstr="%s",
        desc="mathematical operations",
    )
    op1 = File(
        exists=True, mandatory=True, position=-2, argstr="%s", desc="first operator"
    )
    op2 = traits.Either(
        File(exists=True), Str, position=-1, argstr="%s", desc="second operator"
    )
    copy_header = traits.Bool(
        True,
        usedefault=True,
        desc="copy headers of the original image into the output (corrected) file",
    )


class ImageMathOuputSpec(TraitedSpec):
    output_image = File(exists=True, desc="output image file")


class ImageMath(ANTSCommand, CopyHeaderInterface):
    """
    Operations over images.
    Example
    -------
    >>> ImageMath(
    ...     op1='structural.nii',
    ...     operation='+',
    ...     op2='2').cmdline
    'ImageMath 3 structural_maths.nii + structural.nii 2'
    >>> ImageMath(
    ...     op1='structural.nii',
    ...     operation='Project',
    ...     op2='1 2').cmdline
    'ImageMath 3 structural_maths.nii Project structural.nii 1 2'
    >>> ImageMath(
    ...     op1='structural.nii',
    ...     operation='G',
    ...     op2='4').cmdline
    'ImageMath 3 structural_maths.nii G structural.nii 4'
    """

    _cmd = "ImageMath"
    input_spec = ImageMathInputSpec
    output_spec = ImageMathOuputSpec
    _copy_header_map = {"output_image": "op1"}