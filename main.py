from pipeline_digital_library import run_pipeline

result = run_pipeline("data/dbbf8dde-1d40-481b-ba5f-4d84c2de3e54.pdf", pipeline="document", gpu=True, verbose=True)
print(result)