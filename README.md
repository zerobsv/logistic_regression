## Install

=> First, use pyenv to create the virtual environment venv31

$ pyenv activate venv31

=> This script is used to link your local cuda libraries to tensorflow cuda:

$ ./links.sh 


=> Install graphviz on linux, this is required outside of the venv created by pip:

$ sudo apt install graphviz

=> Install all the requirements like tensorflow, scikit learn, and jupyter:

$ pip3 install -r requirements.txt

=> Initialize the jupyter notebook to use the above virtualenv created as the running kernel for the notebook:

$ python3 -m ipykernel install --user --name=venv31 --display-name="Logistic Reg Env"

=> Start jupyter lab and have fun!

$ jupyter lab

