import torch
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, EasyOcrOptions, AcceleratorOptions
)

_converter = None

def get_docling_converter(verbose=True, gpu=True):
    global _converter

    if _converter is None:
        if gpu:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        else:
            device = 'cpu'

        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.ocr_options = EasyOcrOptions(lang=["sk", "en"])
        opts.do_table_structure = False
        opts.images_scale = 2.0
        opts.accelerator_options = AcceleratorOptions(
            device=device,
            num_threads=4
        )

        _converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)
            }
        )

        if verbose:
            print(f"Docling converter loaded on {device.upper()}.")

    return _converter