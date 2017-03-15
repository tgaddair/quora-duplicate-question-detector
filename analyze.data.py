import nltk
from data_helpers import clean_str

max_length = 0
max_sentence = ""

LIMIT = 59
DEV_SIZE = 0.1
TEST_SIZE = 0.1
MAX_SENTENCES = 363916 #40000
TEST_SENTENCES = int(MAX_SENTENCES * TEST_SIZE)

duplicates = 0
sentences = 0

header = True
skipped = 0
skipped_duplicates = []
sentences = 0
with open('quora_duplicate_questions.tsv') as f:
	with open('test.full.tsv', 'w') as fw_test:
		with open('training.full.tsv', 'w') as fw:
			for line in f:
				if header:
					header = False
					continue

				# if sentences >= MAX_SENTENCES + TEST_SENTENCES:
				# 	break

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
						if q1_length > LIMIT:
							skipped_duplicates.append(q1)
						if q2_length > LIMIT:
							skipped_duplicates.append(q2)
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

				if sentences < MAX_SENTENCES:
					fw.write('%s\t%s\t%s\n' % (q1, q2, duplicate))
				else:
					fw_test.write('%s\t%s\t%s\n' % (q1, q2, duplicate))

				print elements
				sentences += 1


print "%s (%d)" % (max_sentence, max_length)
print "duplicates: %d (%.2f)" % (duplicates, ((1.0 * duplicates) / sentences))
print "skipped: %d (%d)" % (skipped, len(skipped_duplicates))
print skipped_duplicates