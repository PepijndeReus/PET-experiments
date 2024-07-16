#!/usr/bin/env bash

# Make directory if it does not exist
mkdir -p raw

# Retrieve datasets
urls=('https://archive.ics.uci.edu/static/public/2/adult.zip' \
      'https://archive.ics.uci.edu/static/public/14/breast+cancer.zip' \
      'https://archive.ics.uci.edu/static/public/320/student+performance.zip' \
          'https://archive.ics.uci.edu/static/public/45/heart+disease.zip')

for url in ${urls[*]}; do
    wget -P raw/ -nc $url;
done

# Unzip files
find 'raw/' -name '*.zip' | while read filename; do unzip -n $filename -d 'raw/'; done;
rm 'raw/.student.zip_old'
unzip -n 'raw/student.zip' -d 'raw/'
