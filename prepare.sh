#!usr/bin/bash
cd $(dirname $0)
python3 loaddata.py && python3 reformdata.py && python3 ner_pico.py