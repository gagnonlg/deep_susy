import re
import matplotlib.pyplot as plt
import argparse

args = argparse.ArgumentParser()
args.add_argument('log')
args.add_argument('name')
args = args.parse_args()

regexp = r'.*loss: (.*) - val_loss: (.*)'

tr_loss = []
va_loss = []
with open(args.log) as logfile:
    for row in logfile:
        match = re.match(regexp, row.strip())
        if match is not None:
            tr_loss.append(float(match.group(1)))
            va_loss.append(float(match.group(2)))

plt.plot(tr_loss, label='training')
plt.plot(va_loss, label='validation')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(args.name)
