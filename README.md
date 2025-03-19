```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/astropy/astropy.git
pyreverse -o dot --source-roots=astropy astropy
python main.py packages.dot [all|<package>] <logfilename>
```