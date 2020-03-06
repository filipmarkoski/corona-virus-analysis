# Coronavirus COVID-19 analysis

* Please read the setup instructions below

# `Tutorials`
1. https://www.kaggle.com/mylesoneill/drosophila-melanogaster-genome/kernels
    1. https://www.kaggle.com/mylesoneill/getting-started-with-biopython

# `Data`
1. `Sequence:` https://www.kaggle.com/paultimothymooney/coronavirus-genome-sequence
2. https://www.kaggle.com/janiscorona/coronavirus-cov-samples-with-sequences-added

# Commands needed to activate the repository:

open `cmd.exe`

before anything else please ensure you're running python `3.6.x` 
* some packages work only with python `3.6`, e.g. `stellargraph`
```
python -V
```
or, in case you have multiple python versions installed
```
python3 -V
```



## Cloning the git repository
clone this repository
```
git clone https://github.com/filipmarkoski/corona-virus-analysis.git
```
go into the folder
```
cd FateMetricWebsite
```
## Creating a virtual environment
create a virtual environment sandbox in the folder named `.env`
```
python -m venv .env
```
activate the virtual environment with:
```
cd .env/Scripts
```

and then by running

```
activate
```

then go back to your root directory `corona-virus-analysis/`
```
cd ../..
```

within your root directory there is a textual file named `requirements.txt`, using it run the following command which will install all needed python packages within the virtual enviroment, i.e. the `.env/` directory's subfolder named `site-packages/`
```
pip install -r requirements.txt
```

wait while the installation finishes.

Done!