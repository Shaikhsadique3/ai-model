.PHONY: install train predict report

install:
	pip install -r requirements.txt

train: install
	python model/train_model.py

predict: install
	python model/predict_model.py --input_csv enhanced_saas_churn_data.csv --output_csv predictions.csv

report: install
	python report/report_generator.py --input_csv predictions.csv --output_pdf report.pdf