import os

with open("CIRCLE_seq_10gRNA_wholeDataset.csv") as fi:
    lines = fi.readlines()

lines = lines[1:]
num = 0
for line in lines:
    items = line.split(",")
    rna1 = items[0]
    rna2 = items[1]
    count = 0
    for i in range(len(rna1)):
        if rna1[i] != rna2[i]:
            count += 1
    if count > 0 and ('_' not in rna1 and '_' in rna2) and items[3] == 'GGTGAGTGAGTGTGTGCGTGNGG':
    # if items[3] == 'GGTGAGTGAGTGTGTGCGTGNGG' and '-' in rna1 and '-' in rna2 and '_' not in rna2:
        num += 1
        # print(count)
        # print(rna1 + "___" + rna2)

print(num)
