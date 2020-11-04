# ResearchProjectAutumn2020
Assessing biases in AI generated human settlement data from high-resolution satellite imagery

# Init environment

```bash
# Create:
conda env create --file environment.yml
# Update (after someone else has changed):
conda env update --file environment.yml
```

Set the `PYTHONPATH` to the repo (module setup):
```bash
export PYTHONPATH="$PYTHONPATH:<location-of-project>/ResearchProjectAutumn2020/src:<location-of-project>/ResearchProjectAutumn2020:<location-of-project>/ResearchProjectAutumn2020/test"
```

# Tests
All tests are located in `test` subdirectory of this project. You can run all suites from `test/runner.py` or single test case from specified test file, e.g. `test/utils/location.py`.
```bash
# In humset conda env
# All suites:
python test/runner.py
# A single test case for a module, e.g.:
python -m unittest test/utils/location.py
```

# Data

Humdata: https://data.humdata.org/dataset/highresolutionpopulationdensitymaps-nga <br>
Worldpop: ftp://ftp.worldpop.org.uk/GIS/Population/Global_2000_2020/2018/NGA/ <br>
GRID3: https://s3-eu-west-1.amazonaws.com/files.grid3.gov.ng/pop/GRID3+-+NGA+-+National+Population+Data+-+v1.2.zip

# Report

To work on the report in *overleaf*, zip the report directory, then in *overleaf* do 
New Project -> Upload Project and select report.zip.

To work on the report locally, you can install the needed tools on Ubuntu by running
```bash
apt install texlive
```
Advised: create separate directory for latex output in the parent of report
```bash
# cwd = report
mkdir ../out
```
To generate pdf run
```bash
# cwd = report
pdflatex --output-directory ../out/ report
```
If references have changed, also need to run
```bash
biber --input-directory ../out/ --output-directory ../out/ report
```
Put all the images in report/images.