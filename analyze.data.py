import nltk
from data_helpers import clean_str

max_length = 0
max_sentence = ""

MIN_LENGTH = 10
LIMIT = 59
DEV_SIZE = 0.1
TEST_SIZE = 0.1

duplicates = 0
sentences = 0

header = True
skipped = 0
skipped_duplicates = 0
sentences = 0

lines = []
with open('quora_duplicate_questions.tsv') as f:
	for line in f:
		if header:
			header = False
			continue

		print line
		elements = line.strip('\n').split('\t')

		q1 = clean_str(elements[3])
		q2 = clean_str(elements[4])
		duplicate = elements[5]

		q1_length = len(q1.split())
		q2_length = len(q2.split())
		
		if q1_length > LIMIT or q2_length > LIMIT:
			skipped += 1
			if duplicate == '1':
				skipped_duplicates += 1
			continue

		if q1_length + q2_length < MIN_LENGTH:
			skipped += 1
			if duplicate == '1':
				skipped_duplicates += 1
			continue

		if q1_length > max_length:
			max_length = q1_length
			max_sentence = q1

		if q2_length > max_length:
			max_length = q2_length
			max_sentence = q2

		if duplicate == '1':
			duplicates += 1

		if len(q1) == 0:
			q1 = "."
		if len(q2) == 0:
			q2 = "."

		lines.append((q1, q2, duplicate))

		print elements
		sentences += 1

test_index = -1 * int(len(lines) * TEST_SIZE) #4000
training_lines = lines[:test_index]
test_lines = lines[test_index:]

with open('training.full.tsv', 'w') as fw:
	for line in training_lines:
		q1, q2, duplicate = line
		fw.write('%s\t%s\t%s\n' % (q1, q2, duplicate))

with open('test.full.tsv', 'w') as fw_test:
	for line in test_lines:
		q1, q2, duplicate = line
		fw_test.write('%s\t%s\t%s\n' % (q1, q2, duplicate))


print "training: ", len(training_lines)
print "test: ", len(test_lines)
print "%s (%d)" % (max_sentence, max_length)
print "duplicates: %d (%.2f)" % (duplicates, ((1.0 * duplicates) / sentences))
print "skipped: %d (%d)" % (skipped, len(skipped_duplicates))