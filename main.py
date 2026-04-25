from pipeline_digital_library import run_pipeline

result = run_pipeline("data/ISLP_website.pdf", pipeline="document", gpu=True, verbose=True)
print(result)