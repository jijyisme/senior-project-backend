#!/usr/bin/env python

import os
import codecs
import optparse
import numpy as np

from tagger import NERTagger

optparser = optparse.OptionParser()

optparser.add_option(
    "-i", "--input", default="",
    help="Input file location"
)
optparser.add_option(
    "-o", "--output", default="",
    help="Output file location"
)

opts = optparser.parse_args()[0]

# Check parameters validity

assert os.path.isfile(opts.input)

tagger = NERTagger()
# y_pred = tagger.predict('data/test.txt')
y_pred = tagger.predict(opts.input)
print('ner tag\n',y_pred)

# Writing output file
f_output = codecs.open(opts.output, 'w', 'utf-8')
for y in y_pred:
	f_output.write(y+'|')
f_output.close()
