from pipeline_digital_library import run_pipeline

result = run_pipeline("data/100_stran_ISLP.pdf", pipeline="document", gpu=True, verbose=True)
print(result)